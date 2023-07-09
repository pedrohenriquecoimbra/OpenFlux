using ContinuousWavelets, Plots, Wavelets
n = 2047;
t = range(0, n / 1000, length=n); # 1kHz sampling rate
f = testfunction(n, "Doppler");
p1 = plot(t, f, legend=false, title="Doppler", xticks=false)
c = wavelet(Morlet(π), β=2);
res = cwt(f, c)
# plotting
freqs = getMeanFreq(computeWavelets(n, c)[1])
freqs[1] = 0
p2 = heatmap(t, freqs, abs.(res)', xlabel="time (s)", ylabel="frequency (Hz)", colorbar=false)
l = @layout [a{0.3h}; b{0.7h}]
plot(p1, p2, layout=l)