import SwiftUI
import IrodoriTTS
import UniformTypeIdentifiers

private enum StudioStyle {
    static let accent = adaptive(light: (0.08, 0.43, 0.39), dark: (0.42, 0.79, 0.71))
    static let buttonInk = adaptive(light: (1, 1, 1), dark: (0.06, 0.15, 0.13))
    static let mutedAccent = Color(red: 0.68, green: 0.87, blue: 0.79)
    static let canvas = adaptive(light: (0.955, 0.963, 0.960), dark: (0.085, 0.095, 0.098))
    static let surface = adaptive(light: (1, 1, 1), dark: (0.125, 0.138, 0.140))
    private static func adaptive(light: (Double, Double, Double), dark: (Double, Double, Double)) -> Color {
        #if os(iOS)
        Color(uiColor: UIColor { traits in
            let value = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: value.0, green: value.1, blue: value.2, alpha: 1)
        })
        #else
        Color(nsColor: NSColor(name: nil) { appearance in
            let value = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: value.0, green: value.1, blue: value.2, alpha: 1)
        })
        #endif
    }
}

struct ContentView: View {
    @StateObject private var model = SampleModel()
    @State private var importing = false
    @State private var importKind: ImportKind = .model
    @State private var downloadExpanded = false
    @State private var renamingVoice: SavedVoice?
    @State private var renameText = ""
    @State private var deletingVoice: SavedVoice?
    @State private var showingVoiceChanger = false
    #if os(macOS)
    @State private var exportingWAV = false
    @State private var wavDocument: SampleWAVDocument?
    #endif
    @FocusState private var focusedField: Field?
    private enum Field { case text, caption, url }
    private enum ImportKind { case model, reference }

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    header
                    if geometry.size.width >= 840 {
                        HStack(alignment: .top, spacing: 24) {
                            workspace.frame(maxWidth: .infinity)
                            settings.frame(width: 310)
                        }
                    } else {
                        workspace
                        settings
                    }
                    Text("Irodori TTS v4.1 Small MF · Community Core ML")
                        .font(.caption2).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                }
                .padding(geometry.size.width >= 840 ? 32 : 20)
                .frame(maxWidth: 1160)
                .frame(maxWidth: .infinity)
            }
            .background(StudioStyle.canvas)
            .scrollDismissesKeyboard(.interactively)
        }
        .tint(StudioStyle.accent)
        .sheet(isPresented: $showingVoiceChanger) { BeatriceView() }
        .fileImporter(isPresented: $importing,
                      allowedContentTypes: importKind == .model ? [.folder] : [.audio]) { result in
            switch result {
            case .success(let url):
                if importKind == .model { model.importModel(url) }
                else { model.importReference(url) }
            case .failure(let error): model.report(error)
            }
        }
        #if os(macOS)
        .fileExporter(isPresented: $exportingWAV, document: wavDocument,
                      contentType: .wav, defaultFilename: "irodori") { result in
            switch result {
            case .success: model.status = "WAVを保存しました。"
            case .failure(let error): model.report(error)
            }
        }
        #endif
        .alert("声の名前を変更", isPresented: Binding(get: { renamingVoice != nil },
                                                set: { if !$0 { renamingVoice = nil } })) {
            TextField("名前", text: $renameText)
            Button("保存") {
                if let voice = renamingVoice { model.renameVoice(voice.id, to: renameText) }
                renamingVoice = nil
            }
            Button("キャンセル", role: .cancel) { renamingVoice = nil }
        }
        .alert("「\(deletingVoice?.name ?? "")」を削除しますか？",
               isPresented: Binding(get: { deletingVoice != nil },
                                    set: { if !$0 { deletingVoice = nil } })) {
            Button("削除", role: .destructive) {
                if let voice = deletingVoice { model.deleteVoice(voice.id) }
                deletingVoice = nil
            }
            Button("キャンセル", role: .cancel) { deletingVoice = nil }
        } message: {
            Text("アプリ内に保存した音声ファイルを削除します。")
        }
        #if os(iOS)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("完了") { focusedField = nil }
            }
        }
        #endif
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(systemName: "waveform")
                .font(.system(size: 25, weight: .medium))
                .foregroundStyle(StudioStyle.accent)
                .frame(width: 54, height: 54)
                .background(StudioStyle.mutedAccent.opacity(0.45), in: RoundedRectangle(cornerRadius: 17))
            VStack(alignment: .leading, spacing: 3) {
                Text("Irodori").font(.system(size: 32, weight: .bold, design: .rounded))
                Text("オンデバイス音声合成").font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button {
                showingVoiceChanger = true
            } label: {
                Label("リアルタイム変声（実験）", systemImage: "waveform.badge.mic")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("openVoiceChanger")
            Text("Core ML")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .padding(.horizontal, 10).padding(.vertical, 7)
                .background(.primary.opacity(0.05), in: Capsule())
        }
    }

    private var workspace: some View {
        VStack(alignment: .leading, spacing: 18) {
            composer
            if let result = model.result {
                metrics(result)
                playback
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Label("読み上げるテキスト", systemImage: "text.alignleft").font(.subheadline.weight(.semibold))
                Spacer()
                Text("\(model.text.count) 文字").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            ZStack(alignment: .topLeading) {
                if model.text.isEmpty {
                    Text("声にしたい文章を入力してください。")
                        .foregroundStyle(.tertiary).padding(.top, 8).padding(.leading, 5)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $model.text)
                    .accessibilityIdentifier("speechText")
                    .font(.system(size: 18)).lineSpacing(7)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 180, maxHeight: 240)
                    .focused($focusedField, equals: .text)
                    .disabled(model.busy || model.recording)
            }
            Rectangle().fill(.primary.opacity(0.07)).frame(height: 1)
            HStack(spacing: 12) {
                Button {
                    focusedField = nil
                    model.speak()
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: "waveform")
                        Text(model.busy ? "処理中…" : "生成して再生")
                    }
                    .font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                }
                .buttonStyle(GenerateButtonStyle())
                .accessibilityIdentifier("speak")
                .disabled(model.busy || model.recording || model.modelPath.isEmpty || model.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if model.busy || model.isPlaying || model.relayActive {
                    Button(action: model.stop) {
                        Image(systemName: "stop.fill").frame(width: 48, height: 48)
                    }
                    .buttonStyle(.bordered).buttonBorderShape(.roundedRectangle(radius: 14))
                    .accessibilityLabel("停止").accessibilityIdentifier("stop")
                }
            }
            Button {
                focusedField = nil
                model.toggleRelay()
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: model.relayActive ? "mic.fill" : "mic")
                    Text(model.relayActive ? "変声モードを終了" : "話して変換（変声モード）")
                }
                .font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.bordered).buttonBorderShape(.roundedRectangle(radius: 14))
            .tint(model.relayActive ? .red : StudioStyle.accent)
            .accessibilityIdentifier("relay")
            .disabled(model.modelPath.isEmpty || model.recording || (model.busy && !model.relayActive))
            if model.relayActive {
                Text(model.heardText.isEmpty ? "（話しかけてください）" : model.heardText)
                    .font(.callout).foregroundStyle(model.heardText.isEmpty ? .tertiary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12).background(StudioStyle.canvas, in: RoundedRectangle(cornerRadius: 12))
                    .accessibilityIdentifier("heardText")
            }
            HStack(alignment: .top, spacing: 9) {
                if model.busy { ProgressView().controlSize(.small) }
                else {
                    Image(systemName: model.errorMessage != nil ? "exclamationmark.circle" : "circle.fill")
                        .font(.system(size: model.errorMessage != nil ? 14 : 6))
                        .padding(.top, model.errorMessage != nil ? 1 : 5)
                }
                Text(model.status).accessibilityIdentifier("status")
                    .font(.caption).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            }
            .foregroundStyle(model.errorMessage != nil ? Color.red : Color.secondary)
        }
        .studioCard()
    }

    private func metrics(_ result: SpeechOutput) -> some View {
        HStack(spacing: 0) {
            metric("RTF", value: String(format: "%.3f", result.rtf))
            Divider().frame(height: 34)
            metric("生成時間", value: String(format: "%.2f s", result.synthesisMilliseconds / 1000))
            Divider().frame(height: 34)
            metric("音声の長さ", value: String(format: "%.2f s", result.audioSeconds))
        }
        .padding(.vertical, 6)
        .accessibilityIdentifier("statistics")
        .accessibilityElement(children: .combine)
    }

    private func metric(_ title: String, value: String) -> some View {
        VStack(spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(size: 23, weight: .semibold, design: .rounded)).monospacedDigit()
                .lineLimit(1).minimumScaleFactor(0.7)
        }.frame(maxWidth: .infinity)
    }

    private var playback: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("生成した音声").font(.subheadline.weight(.semibold))
                Spacer()
                if let url = model.outputURL {
                    #if os(macOS)
                    Button {
                        do {
                            wavDocument = SampleWAVDocument(data: try Data(contentsOf: url))
                            exportingWAV = true
                        } catch { model.report(error) }
                    } label: { Label("WAVを保存", systemImage: "square.and.arrow.down") }
                    .accessibilityIdentifier("saveOutput")
                    .font(.caption.weight(.medium)).disabled(model.busy || model.recording)
                    #else
                    ShareLink(item: url) { Label("WAVを保存", systemImage: "square.and.arrow.up") }
                        .font(.caption.weight(.medium)).disabled(model.busy || model.recording)
                    #endif
                }
            }
            HStack(spacing: 14) {
                Button(action: model.togglePlayback) {
                    Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 17, weight: .semibold))
                        .frame(width: 46, height: 46)
                        .background(StudioStyle.accent.opacity(0.09), in: Circle())
                }
                .buttonStyle(.plain).accessibilityIdentifier("playOutput")
                .accessibilityLabel(model.isPlaying ? "一時停止" : "生成音声を再生")
                .disabled(model.busy || model.recording)
                GeometryReader { geometry in
                    HStack(spacing: 3) {
                        ForEach(Array(model.waveform.enumerated()), id: \.offset) { index, value in
                            Capsule()
                                .fill(Double(index) / Double(max(model.waveform.count, 1)) < model.playbackFraction
                                      ? StudioStyle.accent : StudioStyle.accent.opacity(0.2))
                                .frame(width: max(2, (geometry.size.width - 3 * CGFloat(model.waveform.count - 1)) / CGFloat(max(model.waveform.count, 1))),
                                       height: max(4, CGFloat(value) * 36))
                        }
                    }.frame(height: 40)
                }.frame(height: 40)
                Text(model.playbackTime).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text("AIによる合成音声 · AudioSeal透かし付き · 48 kHz / WAV")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .studioCard()
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 18) {
            modelSettings
            voiceSettings
        }
    }

    private var modelSettings: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("モデル", systemImage: "cpu").font(.subheadline.weight(.semibold))
                Spacer()
                Text(model.modelPath.isEmpty ? "未選択" : model.ready ? "準備済み" : "選択済み")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(model.modelPath.isEmpty ? Color.secondary : StudioStyle.accent)
                    .padding(.horizontal, 9).padding(.vertical, 5)
                    .background(StudioStyle.accent.opacity(0.07), in: Capsule())
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("Irodori v4.1 Small MF").font(.subheadline.weight(.medium))
                Text("Core ML · 約3.0 GB").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button {
                    focusedField = nil; importKind = .model; importing = true
                } label: { Label(model.modelPath.isEmpty ? "フォルダを選ぶ" : "モデルを変更", systemImage: "folder") }
                    .accessibilityIdentifier("importModel")
                Spacer(minLength: 8)
                Button("準備", action: model.prepare).accessibilityIdentifier("prepare")
                    .disabled(model.modelPath.isEmpty)
            }
            .font(.caption.weight(.medium)).buttonStyle(.bordered)
            .disabled(model.busy || model.recording)
            DisclosureGroup("URLからダウンロード", isExpanded: $downloadExpanded) {
                VStack(alignment: .leading, spacing: 10) {
                    TextField("manifest.json のHTTPS URL", text: $model.manifestURL)
                        .textFieldStyle(.roundedBorder).focused($focusedField, equals: .url)
                    Button("ダウンロードして検証", action: model.download)
                        .buttonStyle(.bordered).disabled(model.manifestURL.isEmpty)
                }.padding(.top, 10)
            }
            .font(.caption).disabled(model.busy || model.recording)
            if let milliseconds = model.modelPreparationMilliseconds {
                Text(String(format: "準備 %.0f ms · 参照 %.0f ms%@", milliseconds,
                            model.referencePreparationMilliseconds ?? 0, model.referenceCacheHit ? "（キャッシュ）" : ""))
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("初回の準備には時間がかかることがあります。")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }.studioCard()
    }

    private var voiceSettings: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("声と話し方", systemImage: "person.wave.2").font(.subheadline.weight(.semibold))
                Spacer()
                Text(model.selectedVoice?.name ?? "参照なし")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            TextField("落ち着いた、やさしい話し方。", text: $model.caption, axis: .vertical)
                .accessibilityIdentifier("voiceCaption").lineLimit(2...4).textFieldStyle(.plain)
                .focused($focusedField, equals: .caption)
                .padding(12).background(StudioStyle.canvas, in: RoundedRectangle(cornerRadius: 12))
                .disabled(model.busy || model.recording)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Self.captionPresets, id: \.self) { preset in
                        let selected = model.caption == preset
                        Button {
                            focusedField = nil
                            model.caption = selected ? "" : preset
                        } label: {
                            Text(preset).font(.caption).lineLimit(1)
                                .padding(.horizontal, 11).padding(.vertical, 7)
                                .foregroundStyle(selected ? StudioStyle.buttonInk : Color.primary)
                                .background(selected ? StudioStyle.accent : StudioStyle.accent.opacity(0.08), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .accessibilityIdentifier("captionPresets")
            .disabled(model.busy || model.recording)
            Text("候補をタップすると入力されます。自由に書き換えても、空欄でも生成できます。")
                .font(.caption2).foregroundStyle(.secondary)
            Toggle(isOn: $model.autoTone) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("！や？に合わせて読み分ける").font(.caption.weight(.medium))
                    Text("文末の記号から、文ごとに話し方の指示を足します")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .accessibilityIdentifier("autoTone").toggleStyle(.switch)
            .disabled(model.busy || model.recording)
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                Text("使う声").font(.caption.weight(.medium))
                Picker("使う声", selection: Binding(get: { model.selectedVoiceID },
                                                   set: { model.selectVoice($0) })) {
                    Text("参照なし").tag(UUID?.none)
                    ForEach(model.voices) { voice in
                        Text(voice.name).tag(Optional(voice.id))
                    }
                }
                .pickerStyle(.menu).labelsHidden()
                .accessibilityIdentifier("voicePicker")
                .disabled(model.busy || model.recording)
                if let voice = model.selectedVoice {
                    HStack {
                        Button {
                            renameText = voice.name; renamingVoice = voice
                        } label: { Label("名前を変更", systemImage: "pencil") }
                        Spacer(minLength: 8)
                        Button(role: .destructive) { deletingVoice = voice } label: {
                            Label("この声を削除", systemImage: "trash")
                        }
                        .accessibilityIdentifier("deleteReference")
                    }
                    .font(.caption).buttonStyle(.plain)
                    .disabled(model.busy || model.recording)
                }
            }
            Divider()
            Toggle(isOn: $model.consent) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("利用できる声を登録").font(.caption.weight(.medium))
                    Text("自分の声、または明示的に許可を得た声")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .accessibilityIdentifier("voiceConsent").toggleStyle(.switch)
            .disabled(model.busy || model.recording)
            HStack {
                Button {
                    focusedField = nil; importKind = .reference; importing = true
                } label: { Label("音声を選ぶ", systemImage: "waveform.badge.plus") }
                    .accessibilityIdentifier("importReference")
                    .disabled(model.recording || !model.consent)
                Spacer(minLength: 4)
                Button(action: model.toggleRecording) {
                    Label(model.recording ? "録音を停止" : "録音する", systemImage: model.recording ? "stop.circle" : "mic")
                }
                .accessibilityIdentifier("recordReference")
                .tint(model.recording ? .red : StudioStyle.accent)
                .disabled(!model.recording && !model.consent)
            }
            .font(.caption.weight(.medium)).buttonStyle(.bordered).disabled(model.busy)
            Toggle(isOn: $model.gentleDenoise) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("雑音を弱める（控えめ）").font(.caption.weight(.medium))
                    Text("まわりの音が気になるときだけ。静かな場所ならオフのままがおすすめです")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .accessibilityIdentifier("gentleDenoise").toggleStyle(.switch)
            .disabled(model.busy || model.recording)
            if model.recording { recordingPanel }
            if !model.micDescription.isEmpty && (model.recording || model.pendingReference?.fromRecording == true) {
                Label("マイク: \(model.micDescription)", systemImage: "mic")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let warning = model.micWarning, model.recording || model.pendingReference?.fromRecording == true {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption2).foregroundStyle(.orange)
            }
            if let pending = model.pendingReference, !model.recording { pendingPanel(pending) }
            Text("録音は最初の0.5秒ほど静かにしてから話し始めます。登録した声は上の「使う声」から切り替えられます。")
                .font(.caption2).foregroundStyle(.secondary)
        }.studioCard()
    }

    /// Live guidance while recording: a silent lead-in for the noise profile, then a level meter.
    private var recordingPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "record.circle").foregroundStyle(.red)
                Text(model.recordPhase == .quiet ? "そのまま静かに…" : "話してください")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(String(format: "%.1f 秒", model.recordSeconds))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(levelColor)
                        .frame(width: geometry.size.width * CGFloat(min(1, max(0, (model.inputLevelDB + 60) / 60))))
                }
            }
            .frame(height: 10)
            .accessibilityIdentifier("inputLevel")
            Text(levelText).font(.caption).foregroundStyle(levelColor)
        }
        .padding(12)
        .background(StudioStyle.canvas, in: RoundedRectangle(cornerRadius: 12))
    }

    private var levelText: String {
        if model.recordPhase == .quiet { return "まわりの音を測っています。話さずにお待ちください" }
        switch model.levelHint {
        case .quiet?: return "声が小さいです。もう少し近づくか、はっきり話してください"
        case .loud?: return "大きすぎます。少しマイクから離れてください"
        case .good?: return "ちょうどいい大きさです"
        case nil: return "3〜10秒くらい話して、停止してください"
        }
    }

    private var levelColor: Color {
        if model.recordPhase == .quiet { return .secondary }
        switch model.levelHint {
        case .quiet?: return .orange
        case .loud?: return .red
        default: return StudioStyle.accent
        }
    }

    /// Result of a recording or import: score, reasons, A/B listening and the decision.
    private func pendingPanel(_ pending: PendingReference) -> some View {
        let quality = pending.quality
        let color: Color = quality.grade == .good ? StudioStyle.accent : quality.grade == .fair ? .orange : .red
        let icon = quality.grade == .good ? "checkmark.circle.fill"
            : quality.grade == .fair ? "exclamationmark.circle.fill" : "arrow.counterclockwise.circle.fill"
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(color)
                Text(pending.fromRecording ? "録音の評価: " : "音声の評価: ").font(.subheadline.weight(.semibold))
                    + Text(quality.grade.label).font(.subheadline.weight(.bold)).foregroundColor(color)
            }
            .accessibilityIdentifier("referenceGrade")
            VStack(alignment: .leading, spacing: 4) {
                if quality.reasons.isEmpty {
                    Text("雑音が少なく、声の大きさもちょうどいいです。").font(.caption)
                }
                ForEach(quality.reasons, id: \.self) { reason in
                    Text("・\(reason)").font(.caption).fixedSize(horizontal: false, vertical: true)
                }
            }
            Text(String(format: "使える長さ %.1f 秒 · 声と雑音の差 %.0f dB%@", quality.usableSeconds,
                        max(0, quality.snrDB), pending.noiseReduced ? " · 雑音を弱めています" : ""))
                .font(.caption2).foregroundStyle(.secondary)
            HStack {
                Button { model.togglePreview(.clean) } label: {
                    Label(model.previewing == .clean ? "止める" : "整えた音を聞く",
                          systemImage: model.previewing == .clean ? "stop.fill" : "play.fill")
                }
                .accessibilityIdentifier("previewClean")
                Button { model.togglePreview(.raw) } label: {
                    Label(model.previewing == .raw ? "止める" : "元の音を聞く",
                          systemImage: model.previewing == .raw ? "stop.fill" : "play")
                }
                .accessibilityIdentifier("previewRaw")
            }
            .font(.caption.weight(.medium)).buttonStyle(.bordered)
            HStack {
                Button(action: model.registerPending) {
                    Label("この声で登録", systemImage: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("registerReference")
                .disabled(!quality.hasSpeech || !model.consent)
                Spacer(minLength: 4)
                Button(action: model.retakePending) {
                    Label(pending.fromRecording ? "録り直す" : "取り消す",
                          systemImage: pending.fromRecording ? "arrow.counterclockwise" : "xmark")
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("retakeReference")
            }
            .font(.caption.weight(.medium))
            Text("元の録音も残してあるので、「雑音を弱める」を切り替えると整え直します。")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .disabled(model.busy)
        .padding(12)
        .background(StudioStyle.canvas, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(color.opacity(0.35), lineWidth: 1))
    }

    /// Short Japanese style instructions in the same form as the SDK examples.
    static let captionPresets = [
        "落ち着いた、やさしい話し方。",
        "明るく元気な話し方。",
        "ゆっくり、はっきりとした話し方。",
        "ニュースを読み上げるような、落ち着いた話し方。",
        "ささやくような、静かな話し方。",
        "楽しそうに、笑顔で話す。",
        "悲しそうに、しんみりと話す。",
        "感情を込めて、ドラマチックに話す。",
    ]
}

private struct GenerateButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(enabled ? StudioStyle.buttonInk : Color.secondary)
            .background(enabled ? StudioStyle.accent : Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
            .opacity(configuration.isPressed ? 0.8 : 1)
    }
}

private extension View {
    func studioCard() -> some View {
        padding(22)
            .background(StudioStyle.surface, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(.primary.opacity(0.045), lineWidth: 1))
    }
}

#if os(macOS)
/// Export the completed WAV without changing its header or PCM bytes.
private struct SampleWAVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.wav] }
    let data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.data = data
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
#endif
