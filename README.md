# trng_mic
Ten program generuje prawdziwie losowe liczby (True Random Number Generator - TRNG) wykorzystując następujące elementy: \n
1.Źródło entropii: dźwięk z mikrofonu (szumy otoczenia)
2.Przetwarzanie sygnału: ekstrakcja 3 najmniej znaczących bitów z próbek audio
3.Chaos deterministyczny: zastosowanie układu sprzężonych map namiotowych (Coupled Chaotic Map Lattice - CCML)
4.Perturbacja: mieszanie danych audio z systemem chaotycznym
