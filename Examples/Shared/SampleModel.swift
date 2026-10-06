import SwiftUI
import AVFoundation
import CryptoKit
import Speech
import IrodoriTTS

/// A registered reference voice. Only the file name is persisted; files live in the app's References folder.
struct SavedVoice: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    let fileName: String
}

@MainActor final class SampleModel: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var text = "こんにちは。今日はいい天気なので、近くの公園まで散歩に行きましょう。"
    @Published var caption = UserDefaults.standard.string(forKey: "caption") ?? "" {
        didSet { UserDefaults.standard.set(caption, forKey: "caption") }
    }
    @Published var modelPath = UserDefaults.standard.string(forKey: "modelPath") ?? ""
    @Published private(set) var voices: [SavedVoice] = []
    @Published private(set) var selectedVoiceID: UUID?
    @Published var manifestURL = ""
    @Published var status = "モデルフォルダを選んでください。"
    @Published var statistics = ""
    @Published var busy = false
    @Published var recording = false
    @Published var consent = false
    @Published var outputURL: URL?
    private let engine = IrodoriEngine()
    @Published var result: SynthesisResult?
    @Published var errorMessage: String?
    @Published var ready = false
    @Published var modelPreparationMilliseconds: Double?
    @Published var referencePreparationMilliseconds: Double?
    @Published var referenceCacheHit = false
    @Published var isPlaying = false
    @Published var playbackPosition = 0.0
    @Published var waveform: [Double] = []
    private var player: AVAudioPlayer?
    private var playbackTimer: Timer?
    private var recorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var task: Task<Void, Never>?
    private let store: URL
    // Generate-and-play: chunks are played while later sentences are still being synthesized.
    private var streamPlayer: PCMPlayer?
    private var streamID: UUID?
    // Speak-to-convert mode: on-device speech recognition → synthesis in the selected voice.
    @Published var relayActive = false
    @Published var heardText = ""
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    private let micEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var silenceTimer: Timer?
    /// Silence after the last recognized words that ends an utterance.
    private let endOfSpeechDelay: TimeInterval = 0.7

    override init() {
        store = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("IrodoriSample", isDirectory: true)
        super.init()
        try? FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        loadVoices()
        if !modelPath.isEmpty && !FileManager.default.fileExists(atPath: modelPath) {
            let replacement = store.appendingPathComponent("Models/\(URL(fileURLWithPath: modelPath).lastPathComponent)")
            if FileManager.default.fileExists(atPath: replacement.path) { modelPath = replacement.path }
        }
        if !modelPath.isEmpty {
            // Load Core ML and run one short synthesis at launch so the first real request is not the slow one.
            Task { @MainActor in self.warmUp() }
        }
        if !modelPath.isEmpty {
            status = referencePath.isEmpty
                ? "文章を入力して、音声を生成できます。"
                : "登録音声の利用許可を確認して、音声を生成してください。"
        }
    }

    private func work(_ body: @escaping @MainActor () async throws -> Void) {
        guard !busy, !recording else { return }
        stopPlayback()
        errorMessage = nil
        busy = true
        task = Task {
            defer { busy = false; task = nil; resumeListeningIfIdle() }
            do { try await body() }
            catch is CancellationError { status = "停止しました。" }
            catch { report(error) }
        }
    }

    func importModel(_ source: URL) {
        work {
            self.status = "モデルをコピーして検証しています…"
            let allowed = source.startAccessingSecurityScopedResource()
            defer { if allowed { source.stopAccessingSecurityScopedResource() } }
            let destination = self.store.appendingPathComponent("Models/\(UUID().uuidString)")
            try await Task.detached {
                try ModelBundle.validate(at: source, verifyHashes: true)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                do { try FileManager.default.copyItem(at: source, to: destination) }
                catch { try? FileManager.default.removeItem(at: destination); throw error }
            }.value
            self.stopPlayback()
            await self.engine.release()
            self.ready = false
            self.modelPreparationMilliseconds = nil
            self.referencePreparationMilliseconds = nil
            self.modelPath = destination.path
            UserDefaults.standard.set(destination.path, forKey: "modelPath")
            self.status = "モデルを取り込みました。音声の準備を実行してください。"
        }
    }

    func download() {
        guard let url = URL(string: manifestURL), url.scheme == "https" else { status = "HTTPSのmanifest.json URLを入力してください。"; return }
        work {
            self.status = "モデルを取得しています…"
            let suffix = SHA256.hash(data: Data(url.absoluteString.utf8)).map { String(format: "%02x", $0) }.joined().prefix(24)
            let destination = self.store.appendingPathComponent("Models/download-\(suffix)")
            try await ModelDownloader().download(manifestURL: url, to: destination) { done, total, file in
                Task { @MainActor in self.status = "取得 \(done)/\(total): \(file)" }
            }
            self.stopPlayback()
            await self.engine.release()
            self.ready = false
            self.modelPreparationMilliseconds = nil
            self.referencePreparationMilliseconds = nil
            self.modelPath = destination.path
            UserDefaults.standard.set(destination.path, forKey: "modelPath")
            self.status = "取得と検証が完了しました。"
        }
    }

    func importReference(_ source: URL) {
        guard consent else { status = "利用する声の許可を確認してください。"; return }
        work {
            let allowed = source.startAccessingSecurityScopedResource()
            defer { if allowed { source.stopAccessingSecurityScopedResource() } }
            let destination = self.store.appendingPathComponent("References/\(UUID().uuidString).\(source.pathExtension)")
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try await Task.detached { _ = try ReferenceAudio.read(source); try FileManager.default.copyItem(at: source, to: destination) }.value
            self.addVoice(file: destination, name: source.deletingPathExtension().lastPathComponent)
            self.status = "参照音声を登録しました。"
        }
    }

    func prepare() {
        work { try await self.prepareEngine() }
    }
    private func prepareEngine() async throws {
        guard !modelPath.isEmpty else { throw IrodoriError.invalid("モデルを選択してください。") }
        if !referencePath.isEmpty && !consent { throw IrodoriError.invalid("利用する声の許可を確認してください。") }
        status = "モデルと参照音声を準備しています…"
        let load = try await engine.prepare(modelDirectory: URL(fileURLWithPath: modelPath))
        try Task.checkCancellation()
        let reference = try await engine.registerReference(referencePath.isEmpty ? nil : URL(fileURLWithPath: referencePath))
        try Task.checkCancellation()
        modelPreparationMilliseconds = load
        referencePreparationMilliseconds = reference.milliseconds
        referenceCacheHit = reference.cacheHit
        ready = true
        statistics = String(format: "モデル準備 %.0f ms / 参照 %.0f ms%@", load, reference.milliseconds, reference.cacheHit ? "（キャッシュ）" : "")
        status = "準備できました。"
    }

    func speak() {
        work { try await self.synthesizeAndStream() }
    }

    func warmUp() {
        guard !modelPath.isEmpty else { return }
        work {
            self.status = "起動時の準備をしています…（初回は時間がかかることがあります）"
            let load = try await self.engine.prepare(modelDirectory: URL(fileURLWithPath: self.modelPath))
            try await self.engine.registerReference(nil)
            try Task.checkCancellation()
            _ = try await self.engine.synthesize("こんにちは。", caption: self.caption, splitSentences: true)
            self.modelPreparationMilliseconds = load
            self.ready = self.selectedVoice == nil
            self.status = "準備できました。"
        }
    }

    /// Synthesizes sentence by sentence and plays each PCM chunk as soon as it arrives.
    private func synthesizeAndStream() async throws {
        stopPlayback()
        try await prepareEngine()
        try Task.checkCancellation()
        status = "音声を生成しています…"
        let player = PCMPlayer(managesSession: !relayActive)
        let id = UUID()
        streamPlayer = player; streamID = id
        player.onFinished = { [weak self] in self?.streamDidFinish(id) }
        isPlaying = true
        let result = try await engine.synthesize(text, caption: caption, splitSentences: true) { [weak self] chunk in
            // Main queue keeps chunk order; stale chunks from a stopped request are dropped.
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self, self.streamID == id else { return }
                    do { try player.append(chunk.pcm16) }
                    catch { self.stopPlayback(); self.report(error) }
                }
            }
        }
        try Task.checkCancellation()
        let url = store.appendingPathComponent("generated.wav")
        try result.writeWAV(to: url); outputURL = url
        statistics += String(format: "\nRTF %.3f / 最初のPCM %.0f ms / 音声 %.2f 秒", result.rtf, result.firstPCMMilliseconds, result.audioSeconds)
        self.result = result
        waveform = Self.envelope(result.pcm16)
        playbackPosition = 0
        // Enqueued after every chunk dispatch above, so it runs once all chunks are scheduled.
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                guard self.streamID == id else { return }
                player.finishAppending()
            }
        }
        if streamID == id { status = "再生中です。" }
    }

    private func streamDidFinish(_ id: UUID) {
        guard streamID == id else { return }
        streamPlayer = nil; streamID = nil
        isPlaying = false
        playbackPosition = result?.audioSeconds ?? 0
        status = "再生が完了しました。"
        resumeListeningIfIdle()
    }

    func stop() {
        task?.cancel()
        stopRelay()
        stopPlayback()
        status = busy ? "停止しています…" : "停止しました。"
    }

    var playbackFraction: Double {
        guard let duration = result?.audioSeconds, duration > 0 else { return 0 }
        return min(1, playbackPosition / duration)
    }
    var playbackTime: String {
        let elapsed = Int(playbackPosition)
        let duration = Int(result?.audioSeconds ?? 0)
        return String(format: "%d:%02d / %d:%02d", elapsed / 60, elapsed % 60, duration / 60, duration % 60)
    }

    func togglePlayback() {
        guard !busy, !recording else { return }
        if streamPlayer != nil {
            stopPlayback()
            status = "停止しました。"
            return
        }
        if isPlaying {
            player?.pause()
            isPlaying = false
            playbackTimer?.invalidate(); playbackTimer = nil
            status = "一時停止中です。"
        } else {
            do { try startPlayback() }
            catch { report(error) }
        }
    }

    private func startPlayback() throws {
        guard let url = outputURL else { return }
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio)
        try session.setActive(true)
        #endif
        if player == nil {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
        }
        guard let player else { return }
        if playbackFraction >= 1 { player.currentTime = 0; playbackPosition = 0 }
        guard player.play() else { throw IrodoriError.invalid("音声を再生できませんでした。") }
        errorMessage = nil; isPlaying = true; status = "再生中です。"
        playbackTimer?.invalidate()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isPlaying else { return }
                self.playbackPosition = self.player?.currentTime ?? 0
            }
        }
        playbackTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    private func stopPlayback() {
        streamID = nil
        streamPlayer?.stop(); streamPlayer = nil
        playbackTimer?.invalidate(); playbackTimer = nil
        player?.stop(); player = nil
        isPlaying = false; playbackPosition = 0
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            guard self.player === player else { return }
            self.playbackTimer?.invalidate(); self.playbackTimer = nil
            self.isPlaying = false
            self.playbackPosition = self.result?.audioSeconds ?? 0
            self.status = flag ? "再生が完了しました。" : "音声の再生を完了できませんでした。"
        }
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Task { @MainActor in
            guard self.player === player else { return }
            self.stopPlayback()
            self.report(error ?? IrodoriError.invalid("音声を再生できませんでした。"))
        }
    }

    func report(_ error: Error) {
        let cocoa = error as NSError
        if cocoa.domain == NSCocoaErrorDomain && cocoa.code == NSUserCancelledError { return }
        let message = error.localizedDescription
        status = message.contains("sentence exceeds model limit")
            ? "文章がモデルの上限を超えています。短くして、もう一度生成してください。"
            : message
        errorMessage = status
    }

    // Display actual peak amplitudes from the generated PCM, without changing the audio.
    private static func envelope(_ pcm: Data) -> [Double] {
        let count = pcm.count / 2
        guard count > 0 else { return [] }
        return pcm.withUnsafeBytes { raw in
            (0..<32).map { bin in
                let start = bin * count / 32, end = (bin + 1) * count / 32
                var peak = 0.0
                for index in start..<end {
                    let sample = Int16(littleEndian: raw.loadUnaligned(fromByteOffset: index * 2, as: Int16.self))
                    peak = max(peak, abs(Double(sample)) / 32768)
                }
                return sqrt(peak)
            }
        }
    }

    // MARK: - Voice library

    var selectedVoice: SavedVoice? { voices.first { $0.id == selectedVoiceID } }
    /// Path of the selected voice's audio, or empty when generating without a reference.
    var referencePath: String {
        guard let voice = selectedVoice else { return "" }
        return referencesDirectory.appendingPathComponent(voice.fileName).path
    }
    private var referencesDirectory: URL { store.appendingPathComponent("References", isDirectory: true) }

    func selectVoice(_ id: UUID?) {
        guard !busy, !recording, id != selectedVoiceID else { return }
        selectedVoiceID = id
        ready = false
        saveVoices()
        status = selectedVoice.map { "「\($0.name)」を使います。" } ?? "参照なしで生成します。"
    }

    func renameVoice(_ id: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = voices.firstIndex(where: { $0.id == id }) else { return }
        voices[index].name = trimmed
        saveVoices()
    }

    func deleteVoice(_ id: UUID) {
        guard let voice = voices.first(where: { $0.id == id }) else { return }
        work {
            self.stopPlayback()
            try await self.engine.clearReferenceCache()
            // Delete only this sample's imported/recorded copy, never the selected original.
            let file = self.referencesDirectory.appendingPathComponent(voice.fileName)
            if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
            self.voices.removeAll { $0.id == id }
            if self.selectedVoiceID == id { self.selectedVoiceID = nil }
            self.ready = false
            self.saveVoices()
            self.status = "「\(voice.name)」と参照特徴キャッシュを削除しました。"
        }
    }

    private func addVoice(file: URL, name: String) {
        let base = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let root = base.isEmpty ? "声" : base
        var unique = root
        var suffix = 2
        while voices.contains(where: { $0.name == unique }) { unique = "\(root) \(suffix)"; suffix += 1 }
        let voice = SavedVoice(id: UUID(), name: unique, fileName: file.lastPathComponent)
        voices.append(voice)
        selectedVoiceID = voice.id
        ready = false
        saveVoices()
    }

    private func loadVoices() {
        let defaults = UserDefaults.standard
        if let data = defaults.data(forKey: "voices"),
           let saved = try? JSONDecoder().decode([SavedVoice].self, from: data) {
            voices = saved.filter { FileManager.default.fileExists(atPath: referencesDirectory.appendingPathComponent($0.fileName).path) }
        }
        if let id = defaults.string(forKey: "selectedVoiceID").flatMap(UUID.init(uuidString:)),
           voices.contains(where: { $0.id == id }) {
            selectedVoiceID = id
        }
        // Earlier builds kept a single reference under "referencePath"; carry it over as the first voice.
        if let legacy = defaults.string(forKey: "referencePath") {
            let fileName = URL(fileURLWithPath: legacy).lastPathComponent
            if FileManager.default.fileExists(atPath: referencesDirectory.appendingPathComponent(fileName).path),
               !voices.contains(where: { $0.fileName == fileName }) {
                let voice = SavedVoice(id: UUID(), name: "登録音声 1", fileName: fileName)
                voices.insert(voice, at: 0)
                selectedVoiceID = voice.id
            }
            defaults.removeObject(forKey: "referencePath")
            saveVoices()
        }
    }

    private func saveVoices() {
        let defaults = UserDefaults.standard
        if let data = try? JSONEncoder().encode(voices) { defaults.set(data, forKey: "voices") }
        if let id = selectedVoiceID { defaults.set(id.uuidString, forKey: "selectedVoiceID") }
        else { defaults.removeObject(forKey: "selectedVoiceID") }
    }

    // MARK: - Speak-to-convert mode

    func toggleRelay() {
        if relayActive { stopRelay(); status = "変声モードを終了しました。"; return }
        guard !modelPath.isEmpty else { status = "先にモデルを用意してください。"; return }
        guard selectedVoice == nil || consent else { status = "利用する声の許可を確認してください。"; return }
        guard !busy, !recording else { return }
        relayActive = true
        Task { @MainActor in
            let speechAllowed = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
            }
            let micAllowed = await AVCaptureDevice.requestAccess(for: .audio)
            guard self.relayActive else { return }
            guard speechAllowed, micAllowed else {
                self.relayActive = false
                self.report(IrodoriError.invalid("音声認識とマイクのアクセスを許可してください。"))
                return
            }
            guard self.recognizer?.isAvailable == true else {
                self.relayActive = false
                self.report(IrodoriError.invalid("日本語の音声認識を利用できません。"))
                return
            }
            do { try self.listen() } catch { self.stopRelay(); self.report(error) }
        }
    }

    private func stopRelay() {
        relayActive = false
        stopListening()
    }

    private func resumeListeningIfIdle() {
        guard relayActive, !busy, streamPlayer == nil, recognitionRequest == nil else { return }
        do { try listen() } catch { stopRelay(); report(error) }
    }

    private func listen() throws {
        guard relayActive, !busy, recognitionRequest == nil, let recognizer else { return }
        stopPlayback()
        #if os(iOS)
        // One play-and-record session for the whole mode avoids re-configuring audio between turns.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)
        #endif
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        if recognizer.supportsOnDeviceRecognition { request.requiresOnDeviceRecognition = true }
        let input = micEngine.inputNode
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { buffer, _ in
            request.append(buffer)
        }
        micEngine.prepare()
        try micEngine.start()
        recognitionRequest = request
        heardText = ""
        errorMessage = nil
        status = "聞いています…話し終わると自動で読み上げます。"
        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            let transcript = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self, self.recognitionRequest === request else { return }
                    if let transcript, !transcript.isEmpty {
                        self.heardText = transcript
                        self.armSilenceTimer()
                    }
                    if isFinal || error != nil { self.finishUtterance() }
                }
            }
        }
    }

    private func armSilenceTimer() {
        silenceTimer?.invalidate()
        let timer = Timer(timeInterval: endOfSpeechDelay, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.finishUtterance() }
        }
        silenceTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    private func stopListening() {
        silenceTimer?.invalidate(); silenceTimer = nil
        guard recognitionRequest != nil else { return }
        micEngine.inputNode.removeTap(onBus: 0)
        micEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil; recognitionTask = nil
    }

    /// Ends the current utterance and speaks it; an empty utterance just restarts listening.
    private func finishUtterance() {
        guard recognitionRequest != nil else { return }
        let spoken = heardText.trimmingCharacters(in: .whitespacesAndNewlines)
        stopListening()
        guard relayActive else { return }
        guard !spoken.isEmpty else { resumeListeningIfIdle(); return }
        text = spoken
        work { try await self.synthesizeAndStream() }
    }

    func toggleRecording() {
        if recording {
            recorder?.stop(); recorder = nil; recording = false
            #if os(iOS)
            try? AVAudioSession.sharedInstance().setActive(false)
            #endif
            guard let url = recordingURL else { return }
            do { _ = try ReferenceAudio.read(url)
                let formatter = DateFormatter()
                formatter.dateFormat = "M/d HH:mm"
                addVoice(file: url, name: "録音 \(formatter.string(from: Date()))")
                status = "録音を登録しました。"
            } catch { try? FileManager.default.removeItem(at: url); report(error) }
            return
        }
        guard consent, !busy, !relayActive else { status = "自分の声、または許可のある声を録音してください。"; return }
        work {
            let allowed = await AVCaptureDevice.requestAccess(for: .audio)
            guard allowed else { throw IrodoriError.invalid("マイクのアクセスが許可されていません。") }
            self.stopPlayback()
            #if os(iOS)
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try AVAudioSession.sharedInstance().setActive(true)
            #endif
            let url = self.store.appendingPathComponent("References/\(UUID().uuidString).wav")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            self.recorder = try AVAudioRecorder(url: url, settings: [AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false])
            guard self.recorder?.record() == true else { throw IrodoriError.invalid("録音を開始できません。") }
            self.recordingURL = url; self.recording = true; self.status = "録音中。3〜10秒を目安に停止してください。"
        }
    }
}
