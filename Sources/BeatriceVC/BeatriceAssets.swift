import Foundation

/// `voices.json` written by Beatrice/tools/export_ios_assets.py.
public struct BeatriceManifest: Codable {
    public struct Voice: Codable, Hashable, Identifiable {
        public var id: Int
        public var name: String
        public var file: String
    }
    public struct Common: Codable {
        public var file: String
        public var irWindow: Int
        public var formantShift: [Int]
    }
    public var format: Int
    public var voices: [Voice]
    public var common: Common
    public var functions: [String: Int]
    public var lookaheadFrames: Int
    public var crossfadeFrames: Int
    public var models: [String: String]
    public var credits: [String]
}

/// Speaker-dependent model inputs (one row of the trainer's speaker tables).
public struct BeatriceVoice {
    public var info: BeatriceManifest.Voice
    public var speakerEmbedding: [Float] // 256
    public var keyValue: [Float]         // 384 × 128
    public var codebook: [Float]         // 512 × 128
}

public enum BeatriceError: Error, LocalizedError {
    case missingFile(String)
    case badFile(String)
    case unsupported(String)
    case model(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let s): return "ファイルが見つかりません: \(s)"
        case .badFile(let s): return "ファイルの形式が不正です: \(s)"
        case .unsupported(let s): return s
        case .model(let s): return "モデルの実行に失敗しました: \(s)"
        }
    }
}

/// A directory with voices.json, voice_XXX.bin, common.bin and the .mlpackage models.
/// Swap the directory contents to use speakers trained elsewhere (re-export models and voices together).
public final class BeatriceAssets {
    public let directory: URL
    public let manifest: BeatriceManifest
    public let irWindow: [Float]
    /// 9 × 256, index = round((semitones + 2) · 2) for formant shift −2 … +2 semitones.
    public let formantTable: [Float]

    public init(directory: URL) throws {
        self.directory = directory
        let manifestURL = directory.appendingPathComponent("voices.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw BeatriceError.missingFile(manifestURL.path)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        manifest = try decoder.decode(BeatriceManifest.self, from: data)
        let common = try Self.readFloats(directory.appendingPathComponent(manifest.common.file))
        let formantCount = manifest.common.formantShift.reduce(1, *)
        guard common.count == manifest.common.irWindow + formantCount else {
            throw BeatriceError.badFile(manifest.common.file)
        }
        irWindow = Array(common[0..<manifest.common.irWindow])
        formantTable = Array(common[manifest.common.irWindow...])
    }

    public func loadVoice(_ info: BeatriceManifest.Voice) throws -> BeatriceVoice {
        let values = try Self.readFloats(directory.appendingPathComponent(info.file))
        let e = 256, kv = 384 * 128, cb = 512 * 128
        guard values.count == e + kv + cb else { throw BeatriceError.badFile(info.file) }
        return BeatriceVoice(info: info,
                             speakerEmbedding: Array(values[0..<e]),
                             keyValue: Array(values[e..<(e + kv)]),
                             codebook: Array(values[(e + kv)...]))
    }

    public func modelURL(_ key: String) -> URL? {
        guard let name = manifest.models[key] else { return nil }
        let url = directory.appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Embedding added to the model: speaker row + formant-shift row.
    public func speakerEmbedding(_ voice: BeatriceVoice, formantSemitones: Double) -> [Float] {
        let index = max(0, min(8, Int(((formantSemitones + 2) * 2).rounded())))
        var out = voice.speakerEmbedding
        for i in 0..<256 { out[i] += formantTable[index * 256 + i] }
        return out
    }

    public static func readFloats(_ url: URL) throws -> [Float] {
        guard let data = try? Data(contentsOf: url) else { throw BeatriceError.missingFile(url.path) }
        guard data.count % 4 == 0 else { throw BeatriceError.badFile(url.lastPathComponent) }
        var out = [Float](repeating: 0, count: data.count / 4)
        out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in dst.copyMemory(from: src) }
        }
        return out // files are little-endian float32; all Apple platforms are little-endian
    }
}
