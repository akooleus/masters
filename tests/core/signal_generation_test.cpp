#include "test_utils.h"

int main()
{
    const std::vector<sample_t> noise = generate_white_noise(64, 123);
    const std::vector<sample_t> tones = generate_multitone(64, smoke_spec().fs, {1000.0, 2000.0});
    const std::vector<sample_t> impulse = generate_impulse(16);

    bool ok = true;
    ok &= expect(noise.size() == 64, "white-noise generator preserves requested length");
    ok &= expect(tones.size() == 64, "multitone generator preserves requested length");
    ok &= expect(impulse.size() == 16, "impulse generator preserves requested length");
    ok &= expect(impulse.front() == 1.0f, "impulse begins with one");
    ok &= expect(impulse[1] == 0.0f, "impulse keeps following samples at zero");
    ok &= expect(noise[0] != noise[1], "white-noise generator produces varying samples");
    return ok ? 0 : 1;
}
