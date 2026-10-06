import AVFoundation
import SwiftUI
import BeatriceVC

/// Settings of the voice changer, persisted in UserDefaults.
struct VoiceChangerSettings: Codable, Equatable {
    var quality = "fp32"              // "fp32" | "mixed"
    var computeUnits = BeatriceComputeUnits.cpuAndGPU.rawValue
    var chunkMs = 100
    var function = "w64"              // w64 / w116 / w216
    var voiceID: Int?
    var pitchShift = 0                // semitones
    var formant = 0.0                 // semitones, -2 ... 2
    var useSpeaker = false
    var noiseCut = 0.3                // 0 = off ... 1 = strong
    var micLowCut = true
    var outputLowCut = true
    var voiceProcessing = false

    static let key = "BeatriceVoiceChangerSettings.v1"

    static func load() -> VoiceChangerSettings {
        guard let data = UserDefaults.standard.data(forKey: key),
              let s = try? JSONDecoder().decode(VoiceChangerSettings.self, from: data) else { return .init() }
        return s
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }

    /// noise-cut strength -> gate threshold (dBFS); nil = off
    var gateThresholdDB: Float? {
        noiseCut < 0.02 ? nil : Float(-80 + 50 * noiseCut)
    }

    static func noiseCut(forThreshold db: Float) -> Double {
        min(1, max(0.05, Double(db + 80) / 50))
    }
}

/// 「リアルタイム変声（実験）」: Beatrice v2 pseudo-streaming voice conversion on device.
@MainActor final class BeatriceViewModel: ObservableObject {
    @Published var settings = VoiceChangerSettings.load() {
        didSet {
            settings.save()
            if running, settings.noiseCut != oldValue.noiseCut {
                engine.setGateThreshold(settings.gateThresholdDB)
            }
        }
    }
    @Published private(set) var running = false
    @Published private(set) var preparing = false
    @Published private(set) var measuring = false
    @Published private(set) var status = ""
    @Published private(set) var stats = BeatriceRealtimeEngine.Stats()
    @Published private(set) var voices: [BeatriceManifest.Voice] = []
    @Published private(set) var credits: [String] = []

    private var assets: BeatriceAssets?
    private let engine = BeatriceRealtimeEngine()
    private var timer: Timer?

    init() {
        do {
            guard let dir = Bundle.main.url(forResource: "BeatriceAssets", withExtension: nil) else {
                throw BeatriceError.missingFile("BeatriceAssets")
            }
            let assets = try BeatriceAssets(directory: dir)
            self.assets = assets
            voices = assets.manifest.voices
            credits = assets.manifest.credits
            if settings.voiceID == nil || !voices.contains(where: { $0.id == settings.voiceID }) {
                settings.voiceID = voices.first?.id
            }
        } catch {
            status = error.localizedDescription
        }
    }

    var settingsLocked: Bool { running || preparing }

    func toggle() {
        if running { stop() } else { start() }
    }

    func stop() {
        engine.stop()
        timer?.invalidate(); timer = nil
        running = false
        stats = engine.snapshot()
        status = "停止しました。"
    }

    func resetSettings() {
        let voice = settings.voiceID
        settings = VoiceChangerSettings()
        settings.voiceID = voice
    }

    private func start() {
        let s = settings
        guard let assets, let voiceID = s.voiceID, let info = voices.first(where: { $0.id == voiceID }) else { return }
        guard #available(iOS 18.0, macOS 15.0, *) else {
            status = "この機能は iOS 18 / macOS 15 以降が必要です。"
            return
        }
        guard let url = assets.modelURL(s.quality), let window = assets.manifest.functions[s.function] else {
            status = "モデルが同梱されていません。"
            return
        }
        preparing = true
        status = "準備しています…（初回は 10 秒ほどかかることがあります）"
        let units = BeatriceComputeUnits(rawValue: s.computeUnits) ?? .cpuAndGPU
        Task {
            let granted = await AVAudioApplicationPermission.request()
            guard granted else {
                preparing = false
                status = "マイクの使用が許可されていません。設定アプリで許可してください。"
                return
            }
            do {
                let streamer = try await Task.detached(priority: .userInitiated) { () throws -> BeatriceStreamer in
                    let model = try BeatriceModel(package: url, function: s.function, window: window, computeUnits: units)
                    let voice = try assets.loadVoice(info)
                    try model.setSpeaker(embedding: assets.speakerEmbedding(voice, formantSemitones: s.formant),
                                         keyValue: voice.keyValue, codebook: voice.codebook)
                    let st = BeatriceStreamer(predictor: model, dsp: BeatriceDSP(irWindow: assets.irWindow),
                                              chunk: s.chunkMs / 10, lookahead: assets.manifest.lookaheadFrames,
                                              crossfade: assets.manifest.crossfadeFrames)
                    st.pitchShiftBins = Float(s.pitchShift * 8)
                    // warm-up (the first prediction is slow)
                    _ = try st.process([Float](repeating: 0, count: st.inputChunkSamples))
                    st.inputHighPass = s.micLowCut ? BeatriceHighPass(cutoff: 75, sampleRate: 16_000) : nil
                    st.outputHighPass = s.outputLowCut ? BeatriceHighPass(cutoff: 50, sampleRate: 24_000) : nil
                    st.gate = s.gateThresholdDB.map { BeatriceNoiseGate(thresholdDB: $0) }
                    return st
                }.value
                try engine.start(streamer: streamer, useSpeaker: s.useSpeaker, voiceProcessing: s.voiceProcessing)
                running = true
                status = "変換中です。"
                timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
                    Task { @MainActor in self?.refresh() }
                }
            } catch {
                status = error.localizedDescription
            }
            preparing = false
        }
    }

    func measureNoise() {
        guard running, !measuring else { return }
        measuring = true
        status = "周りの雑音を測っています。2 秒間、声を出さずにお待ちください…"
        Task {
            if let floor = await engine.measureNoiseFloor(seconds: 2) {
                settings.noiseCut = VoiceChangerSettings.noiseCut(forThreshold: floor + 6)
                status = String(format: "周りの雑音は約 %.0f dB でした。雑音カットを合わせました。", floor)
            } else {
                status = "測れませんでした。変換中にもう一度お試しください。"
            }
            measuring = false
        }
    }

    func recordDiagnostics() {
        guard running else { return }
        status = "診断用に 5 秒間録音しています。普段どおり話してください…"
        Task {
            do {
                let dir = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                let files = try await engine.recordDiagnostics(seconds: 5, to: dir)
                status = "保存しました（「ファイル」アプリ → このアプリ）: " + files.map(\.lastPathComponent).joined(separator: ", ")
            } catch {
                status = error.localizedDescription
            }
        }
    }

    private func refresh() {
        stats = engine.snapshot()
        if let e = stats.lastError { status = "エラー: \(e)" }
    }
}

enum AVAudioApplicationPermission {
    static func request() async -> Bool {
        await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) }
        }
    }
}

struct BeatriceView: View {
    @StateObject private var model = BeatriceViewModel()
    @Environment(\.dismiss) private var dismiss
    @State private var showDetails = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label("有線イヤホンをおすすめします。スピーカーから出すとハウリングすることがあり、Bluetooth イヤホンは遅れが大きくなります。",
                          systemImage: "headphones")
                        .font(.callout)
                    Text("実験中の機能です。声の変換にはしばらく前からの音をまとめて使うため、少し遅れて聞こえます。")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("声") {
                    Picker("話者", selection: $model.settings.voiceID) {
                        ForEach(model.voices) { Text($0.name).tag(Optional($0.id)) }
                    }
                    Stepper(value: $model.settings.pitchShift, in: -12...12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("声の高さ：\(pitchText)")
                            Text("1 目盛りで半音。12 で 1 オクターブ").font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    Stepper(value: $model.settings.formant, in: -2...2, step: 0.5) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("声の太さ：\(formantText)")
                            Text("マイナスで太く大人っぽく、プラスで細く若々しく（目安）").font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
                .disabled(model.settingsLocked)

                Section {
                    VStack(alignment: .leading) {
                        Text("雑音カットの強さ：\(noiseCutText)")
                        Slider(value: $model.settings.noiseCut, in: 0...1)
                        Text("声を出していない間の音を消します。強くしすぎると、小さな声や声の出だしも消えます。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Button(model.measuring ? "測っています…" : "周りの雑音を測って合わせる（2 秒静かに）") {
                        model.measureNoise()
                    }
                    .disabled(!model.running || model.measuring)
                    if !model.running {
                        Text("雑音の測定は変換中に使えます。").font(.caption2).foregroundStyle(.secondary)
                    }
                    Group {
                        Toggle("マイクの低い雑音（空調・振動）をカット", isOn: $model.settings.micLowCut)
                        Toggle("変換した声の低いうなりをカット", isOn: $model.settings.outputLowCut)
                        Toggle(isOn: $model.settings.voiceProcessing) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("端末の雑音除去を使う")
                                Text("遅れが増えたり、声の感じが変わることがあります。").font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .disabled(model.settingsLocked)
                } header: {
                    Text("雑音対策")
                }

                Section("音質と速さ") {
                    Picker("音質", selection: $model.settings.quality) {
                        Text("高音質").tag("fp32")
                        Text("軽量（電池・発熱に優しい）").tag("mixed")
                    }
                    Picker("計算に使う部分", selection: $model.settings.computeUnits) {
                        Text("CPU").tag(BeatriceComputeUnits.cpuOnly.rawValue)
                        Text("GPU（おすすめ）").tag(BeatriceComputeUnits.cpuAndGPU.rawValue)
                        Text("Neural Engine").tag(BeatriceComputeUnits.cpuAndNeuralEngine.rawValue)
                        Text("自動").tag(BeatriceComputeUnits.all.rawValue)
                    }
                    Picker("反応の速さ", selection: $model.settings.chunkMs) {
                        Text("速い（50ms）").tag(50)
                        Text("安定（100ms）").tag(100)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Picker("なめらかさ", selection: $model.settings.function) {
                            Text("短い（0.5 秒）").tag("w64")
                            Text("ふつう（1 秒）").tag("w116")
                            Text("長い（2 秒）").tag("w216")
                        }
                        Text("変換のときに参照する直前の音の長さ。長いほど声が安定しますが、計算に時間がかかります。")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    #if os(iOS)
                    Toggle("内蔵スピーカーから出す", isOn: $model.settings.useSpeaker)
                    #endif
                    Button("設定を最初の状態に戻す") { model.resetSettings() }
                        .font(.caption)
                }
                .disabled(model.settingsLocked)

                Section {
                    Button {
                        model.toggle()
                    } label: {
                        Label(model.running ? "停止" : (model.preparing ? "準備中…" : "開始"),
                              systemImage: model.running ? "stop.fill" : "mic.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(model.preparing || model.settings.voiceID == nil)
                    if !model.status.isEmpty {
                        Text(model.status).font(.caption).foregroundStyle(.secondary)
                    }
                }

                Section("状態") {
                    row("1 回の変換時間", String(format: "%.0f ms %@", model.stats.inferenceMedianMs, speedNote))
                    row("音の途切れ", "\(model.stats.droppedChunks + model.stats.underruns) 回")
                    row("遅れ（目安）", String(format: "約 %.0f ms", model.stats.estimatedLatencyMs))
                    row("入力の大きさ", inputLevelText)
                    DisclosureGroup("詳細", isExpanded: $showDetails) {
                        detail("変換時間（中央値 / 最大）", String(format: "%.1f / %.1f ms", model.stats.inferenceMedianMs, model.stats.inferenceMaxMs))
                        detail("処理したまとまり", "\(model.stats.chunks)")
                        detail("入力の破棄（処理が追いつかない）", "\(model.stats.droppedChunks)")
                        detail("出力の不足", String(format: "%d 回（計 %.0f ms）", model.stats.underruns, model.stats.underrunMs))
                        detail("時刻の飛び（入力 / 出力）", "\(model.stats.inputTimestampJumps) / \(model.stats.renderTimestampJumps)")
                        detail("1 回の受け渡し（入力 / 出力）", "\(model.stats.inputCallbackFrames) / \(model.stats.renderCallbackFrames) サンプル")
                        detail("マイク", String(format: "%.0f Hz・%d ch", model.stats.inputSampleRate, model.stats.inputChannels))
                        detail("出力", String(format: "%.0f Hz", model.stats.outputSampleRate))
                        detail("I/O バッファ", String(format: "%.1f ms（%.0f Hz）", model.stats.ioBufferDuration * 1000, model.stats.hardwareSampleRate))
                        detail("機器の遅れ（入力 / 出力）", String(format: "%.1f / %.1f ms", model.stats.inputLatency * 1000, model.stats.outputLatency * 1000))
                        detail("処理単位 / 参照 / 先読み", String(format: "%.0f / %.0f / %.0f ms", model.stats.chunkMs, model.stats.contextMs, model.stats.lookaheadMs))
                        detail("再生前のため込み", String(format: "%.0f ms", model.stats.prebufferMs))
                        detail("端末の雑音除去", model.stats.voiceProcessing ? "オン" : "オフ")
                        Text("遅れ（目安）= I/O バッファ×2 ＋ 機器の遅れ ＋ 処理単位 ＋ ため込み ＋ 先読み 40ms ＋ 変換時間")
                            .font(.caption2).foregroundStyle(.secondary)
                        Button("診断用に 5 秒録音して保存") { model.recordDiagnostics() }
                            .disabled(!model.running)
                            .font(.caption)
                    }
                    .font(.caption)
                }

                Section("クレジット") {
                    ForEach(model.credits, id: \.self) { Text($0).font(.caption) }
                    Text("話者は事前学習モデルに含まれる話者を番号で表示しています（LibriTTS-R 由来、CC BY 4.0）。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .navigationTitle("リアルタイム変声（実験）")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("閉じる") {
                        model.stop()
                        dismiss()
                    }
                }
            }
        }
        .onDisappear { if model.running { model.stop() } }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 640)
        #endif
    }

    private var pitchText: String {
        let p = model.settings.pitchShift
        return p == 0 ? "元のまま" : (p > 0 ? "\(p) 高く" : "\(-p) 低く")
    }

    private var formantText: String {
        let f = model.settings.formant
        if f == 0 { return "ふつう" }
        return f < 0 ? String(format: "太く（%.1f）", -f) : String(format: "細く（%.1f）", f)
    }

    private var noiseCutText: String {
        let v = model.settings.noiseCut
        if v < 0.02 { return "切" }
        return v < 0.35 ? "弱" : (v < 0.65 ? "中" : "強")
    }

    private var speedNote: String {
        guard model.stats.chunks > 0 else { return "" }
        return model.stats.inferenceMedianMs < model.stats.chunkMs ? "（間に合っています）" : "（遅れています）"
    }

    private var inputLevelText: String {
        guard model.running else { return "—" }
        let state = model.stats.gateThresholdDB == nil ? "" : (model.stats.gateOpen ? "・声あり" : "・待機中")
        return String(format: "%.0f dB", model.stats.inputLevelDB) + state
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    private func detail(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).font(.caption)
            Spacer()
            Text(value).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }
}
