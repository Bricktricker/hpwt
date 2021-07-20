# Hybride parallele Wavelet Tree Konstruktion

Code meiner Bachelorarbeit zum hybriden Konstruiren von Wavelet Trees. Die Konstruktion geschieht mithilfe einer Domain Decomposition. Dafür wird MPI und openMP verwendet. Basiert auf [https://github.com/pdinklag/distwt/](https://github.com/pdinklag/distwt/) und [https://github.com/kurpicz/pwm/](https://github.com/kurpicz/pwm/).  
Für die Theoretischen Grundlagen siehe [*Constructing the Wavelet Tree and Wavelet Matrix in Distributed Memory*](https://doi.org/10.1137/1.9781611976007.17) [Dinklage et al., ALENEX 2020] und [*Simple, Fast and Lightweight Parallel Wavelet Tree Construction*](https://arxiv.org/abs/1702.07578)[Kurpicz et al.]

## Bauen
Das Projekt ist ein CMake Projekt, welches einfach mit
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
``` 
gebaut werden kann. Dabei werden zwei Programme gebaut, welche unter `build/src/` gefunden werden können. Einmal `hpwt_ppc`, welches den Parallel Prefix Counting Algorithmus zur lokalen Konstruktion verwendet. Desweiteren wird das `hpwt_pps` Programm gebaut, welches einen Parallel Prefix Sorting Algorithmus zur lokalen Konstruktion verwendet. Das `hpwt_ppc` Programm war in allen Test schneller und sollte immer verwendet werden.

## Abhänigkeiten
Neben MPI und openMP wird noch [tlx](https://github.com/tlx/tlx) benötigt, welches manuell zuerst installiert werden muss.

## Nutzung
- Als minimales Argument wird die Eingabedatei erwartet, für welche der Wavelet Tree berechnet werden soll.
- Soll der berechnet Baum auch gespeichert werden, kann `-o /path/to/file` als  Argument genutzt werden.
- Des Weiteren ist es möglich nur einen Präfix der Eingabe zu verwenden, dieser kann mit `-p 1Gi` angegeben werden, um z.B. nur das erste Gibibyte der Eingabe zu verwenden.
- Mit `-w 1` kann angegeben werden wie viele Bytes pro Eingabezeichen verwendet werden sollen. Valide Größen sind `1, 2, 4, 5`. 
- Sollte der finale Wavelet Tree überprüft werden, ob dieser korrekt konstruiert wurde, kann das Programm mit `-v` gestartet werden. Dabei ist es notwendig den Wavelet Tree vorher zu Speichern, also das Programm mit `-o` zu starten.
- Des Weiteren kann mit `-r X` festgelegt werden, in wievielen Byte Blöcken die Eingabe gelesen werden soll. Wobei `X` eine valide Größe wie `1Gi` ist.
