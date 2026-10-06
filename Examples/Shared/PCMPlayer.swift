import Foundation
import AVFoundation

@MainActor final class PCMPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 1, interleaved: false)!
    private let managesSession: Bool
    private var pending = 0
    private var appendingFinished = false
    /// Called once on the main actor after `finishAppending()` and every scheduled buffer has been played.
    var onFinished: (@MainActor () -> Void)?

    /// Pass `managesSession: false` when the caller keeps its own AVAudioSession (e.g. play-and-record).
    init(managesSession: Bool = true) {
        self.managesSession = managesSession
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }
    func append(_ data: Data) throws {
        if !engine.isRunning && managesSession {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio)
            try session.setActive(true)
            #endif
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(data.count / 2)),
              let samples = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = buffer.frameCapacity
        data.withUnsafeBytes { raw in
            for i in 0..<Int(buffer.frameLength) {
                let value = raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self)
                samples[i] = Float(Int16(littleEndian: value)) / 32768
            }
        }
        if !engine.isRunning { try engine.start() }
        pending += 1
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self?.bufferPlayed() }
            }
        }
        if !node.isPlaying { node.play() }
    }
    /// No more chunks will be appended; `onFinished` fires when the queued audio has played.
    func finishAppending() {
        appendingFinished = true
        notifyIfFinished()
    }
    func stop() {
        onFinished = nil
        node.stop(); engine.stop()
    }
    private func bufferPlayed() {
        pending = max(0, pending - 1)
        notifyIfFinished()
    }
    private func notifyIfFinished() {
        guard appendingFinished, pending == 0, let callback = onFinished else { return }
        onFinished = nil
        callback()
    }
}
