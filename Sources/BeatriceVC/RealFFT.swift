import Accelerate
import Foundation

/// Real-input DFT of length `n` (n = f · 2^k, f ∈ {1, 3, 5, 15}) with plain DFT semantics:
///
/// * `forward`: X[k] = Σ_t x[t] · e^{-2πikt/n}, k = 0 … n/2 (input shorter than n is zero-padded)
/// * `inverse`: numpy/torch `irfft` (imaginary parts of the DC and Nyquist bins are ignored)
///
/// Backed by `vDSP_DFT_zrop`. Its packing (DC in `realp[0]`, Nyquist in `imagp[0]`) and scale
/// factors are calibrated once at init so callers never see vDSP's conventions.
public final class RealFFT {
    public let n: Int
    public var bins: Int { n / 2 + 1 }
    private let forwardSetup: vDSP_DFT_Setup
    private let inverseSetup: vDSP_DFT_Setup
    private var forwardScale: Float = 1
    private var inverseScale: Float = 1
    private var inR: [Float], inI: [Float], outR: [Float], outI: [Float]

    public init?(n: Int) {
        guard n >= 16, n % 2 == 0,
              let f = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(n), .FORWARD),
              let i = vDSP_DFT_zrop_CreateSetup(f, vDSP_Length(n), .INVERSE) else { return nil }
        self.n = n
        forwardSetup = f
        inverseSetup = i
        let half = n / 2
        inR = [Float](repeating: 0, count: half); inI = inR; outR = inR; outI = inR
        // Calibrate: the DFT of a unit impulse is 1 in every bin; the irfft of an all-ones
        // spectrum is a unit impulse.
        var impulse = [Float](repeating: 0, count: n); impulse[0] = 1
        var re = [Float](repeating: 0, count: bins), im = re
        forward(impulse, &re, &im)
        forwardScale = 1 / re[1]
        let ones = [Float](repeating: 1, count: bins)
        let zeros = [Float](repeating: 0, count: bins)
        var out = [Float](repeating: 0, count: n)
        inverse(ones, zeros, &out)
        inverseScale = 1 / out[0]
    }

    deinit {
        vDSP_DFT_DestroySetup(forwardSetup)
        vDSP_DFT_DestroySetup(inverseSetup)
    }

    /// `x` may be shorter than `n` (zero-padded). `re`/`im` receive `bins` values.
    public func forward(_ x: [Float], _ re: inout [Float], _ im: inout [Float]) {
        x.withUnsafeBufferPointer { forward($0.baseAddress!, count: x.count, &re, &im) }
    }

    public func forward(_ x: UnsafePointer<Float>, count: Int, _ re: inout [Float], _ im: inout [Float]) {
        let half = n / 2
        let m = min(count, n)
        for t in 0..<half {
            let a = 2 * t, b = 2 * t + 1
            inR[t] = a < m ? x[a] : 0
            inI[t] = b < m ? x[b] : 0
        }
        vDSP_DFT_Execute(forwardSetup, inR, inI, &outR, &outI)
        if re.count < bins { re = [Float](repeating: 0, count: bins) }
        if im.count < bins { im = [Float](repeating: 0, count: bins) }
        let s = forwardScale
        re[0] = outR[0] * s; im[0] = 0
        re[half] = outI[0] * s; im[half] = 0
        for k in 1..<half {
            re[k] = outR[k] * s
            im[k] = outI[k] * s
        }
    }

    /// `re`/`im` hold `bins` values; `out` receives `n` samples.
    public func inverse(_ re: [Float], _ im: [Float], _ out: inout [Float]) {
        let half = n / 2
        inR[0] = re[0]
        inI[0] = re[half]
        for k in 1..<half {
            inR[k] = re[k]
            inI[k] = im[k]
        }
        vDSP_DFT_Execute(inverseSetup, inR, inI, &outR, &outI)
        if out.count < n { out = [Float](repeating: 0, count: n) }
        let s = inverseScale
        for t in 0..<half {
            out[2 * t] = outR[t] * s
            out[2 * t + 1] = outI[t] * s
        }
    }
}
