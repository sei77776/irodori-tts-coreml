import AVFoundation
import SwiftUI
import BeatriceVC

/// 「リアルタイム変声（実験）」: Beatrice v2 pseudo-streaming voice conversion on device.
@MainActor final class BeatriceViewModel: ObservableObject {
    enum Precision: String, CaseIterable, Identifiable {
        case fp32 = "高精度 (FP32)"
        case mixed = "軽量 (FP16 混在)"
        var id: String { rawValue }
        var key: String { self == .fp32 ? "fp32" : "mixed" }
    }
    enum Context: String, CaseIterable, Identifiable {
        case half = "0.5 秒", one = "1 秒", two = "2 秒"
        var id: String { rawValue }
        var function: String { switch self { case .half: return "w64"; case .one: return "w116"; case .two: return "w216" } }
    }

    @Published var precision: Precision = .fp32
    @Published var context: Context = .half
    @Published var chunkMs = 100
    @Published var units: BeatriceComputeUnits = .all
    @Published var voiceID: Int?
    @Published var pitchShift = 0
    @Published var formant = 0.0
    @Published var useSpeaker = false
    @Published private(set) var running = false
    @Published private(set) var preparing = false
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
            voiceID = voices.first?.id
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

    private func start() {
        guard let assets, let voiceID, let info = voices.first(where: { $0.id == voiceID }) else { return }
        guard #available(iOS 18.0, macOS 15.0, *) else {
            status = "この機能は iOS 18 / macOS 15 以降が必要です（Core ML の複数関数モデルを使うため）。"
            return
        }
        guard let url = assets.modelURL(precision.key), let window = assets.manifest.functions[context.function] else {
            status = "モデルが同梱されていません。"
            return
        }
        preparing = true
        status = "モデルを準備しています…（初回はコンパイルに時間がかかります）"
        let function = self.context.function, computeUnits = self.units, chunk = self.chunkMs / 10
        let pitch = Float(self.pitchShift * 8), formantShift = self.formant, speaker = self.useSpeaker
        Task {
            let granted = await AVAudioApplicationPermission.request()
            guard granted else {
                preparing = false
                status = "マイクの使用が許可されていません。設定アプリで許可してください。"
                return
            }
            do {
                let streamer = try await Task.detached(priority: .userInitiated) { () throws -> BeatriceStreamer in
                    let model = try BeatriceModel(package: url, function: function, window: window, computeUnits: computeUnits)
                    let voice = try assets.loadVoice(info)
                    try model.setSpeaker(embedding: assets.speakerEmbedding(voice, formantSemitones: formantShift),
                                         keyValue: voice.keyValue, codebook: voice.codebook)
                    let s = BeatriceStreamer(predictor: model, dsp: BeatriceDSP(irWindow: assets.irWindow),
                                             chunk: chunk, lookahead: assets.manifest.lookaheadFrames,
                                             crossfade: assets.manifest.crossfadeFrames)
                    s.pitchShiftBins = pitch
                    // warm-up (first prediction is slow)
                    _ = try s.process([Float](repeating: 0, count: s.inputChunkSamples))
                    return s
                }.value
                try engine.start(streamer: streamer, useSpeaker: speaker)
                running = true
                status = "変換中（左文脈 \(streamer.context * 10) ms・チャンク \(streamer.chunk * 10) ms・先読み \(streamer.lookahead * 10) ms）"
                timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                    Task { @MainActor in self?.refresh() }
                }
            } catch {
                status = error.localizedDescription
            }
            preparing = false
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

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label("有線イヤホンを推奨します。スピーカー出力ではハウリングすることがあります。Bluetooth は遅延が大きくなります。",
                          systemImage: "headphones")
                        .font(.callout)
                    Text("実験機能です。Beatrice v2 の事前学習モデルを窓をずらしながら毎回まるごと推論する簡易方式（疑似ストリーミング）で動かしています。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("声") {
                    Picker("話者", selection: $model.voiceID) {
                        ForEach(model.voices) { Text($0.name).tag(Optional($0.id)) }
                    }
                    Stepper("ピッチ \(model.pitchShift >= 0 ? "+" : "")\(model.pitchShift) 半音", value: $model.pitchShift, in: -12...12)
                    Stepper(String(format: "フォルマント %+.1f 半音", model.formant), value: $model.formant, in: -2...2, step: 0.5)
                }
                .disabled(model.settingsLocked)
                Section("処理") {
                    Picker("モデル", selection: $model.precision) {
                        ForEach(BeatriceViewModel.Precision.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Picker("チャンク", selection: $model.chunkMs) {
                        Text("50 ms").tag(50); Text("100 ms").tag(100)
                    }
                    Picker("左文脈", selection: $model.context) {
                        ForEach(BeatriceViewModel.Context.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Picker("Compute units", selection: $model.units) {
                        ForEach(BeatriceComputeUnits.allCases) { Text($0.rawValue).tag($0) }
                    }
                    #if os(iOS)
                    Toggle("内蔵スピーカーから出す", isOn: $model.useSpeaker)
                    #endif
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
                    .disabled(model.preparing || model.voiceID == nil)
                    if !model.status.isEmpty {
                        Text(model.status).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("計測") {
                    stat("推論時間（中央値 / 最大）", String(format: "%.1f / %.1f ms", model.stats.inferenceMedianMs, model.stats.inferenceMaxMs))
                    stat("処理したチャンク", "\(model.stats.chunks)")
                    stat("処理落ち（入力破棄 / 出力途切れ）", "\(model.stats.droppedChunks) / \(model.stats.underruns)")
                    stat("I/O バッファ", String(format: "%.1f ms（%.0f Hz）", model.stats.ioBufferDuration * 1000, model.stats.hardwareSampleRate))
                    stat("入出力の機器遅延", String(format: "%.1f / %.1f ms", model.stats.inputLatency * 1000, model.stats.outputLatency * 1000))
                    stat("推定遅延", String(format: "約 %.0f ms", model.stats.estimatedLatencyMs))
                    Text("推定遅延 = I/O バッファ×2 + 機器遅延 + チャンク×2（処理単位＋出力の揺らぎ吸収）+ 先読み 40 ms + 推論時間")
                        .font(.caption2).foregroundStyle(.secondary)
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

    private func stat(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).font(.caption)
            Spacer()
            Text(value).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }
}
