# Wie der Datensatz entstanden ist

**Sprache:** [English](dataset-creation.md) · [فارسی](dataset-creation.fa.md) · Deutsch

---

Der Trainingsdatensatz dieses Projekts, 300 parallele Frage Antwort Paare auf Persisch,
Englisch und Deutsch, wurde mit **Claude Code unter Opus 5 bei besonders hohem
Reasoning Aufwand** recherchiert, verfasst, strukturiert und validiert.

Diese Seite dokumentiert das Vorgehen und sein Ergebnis, denn der Datensatz ist das
stärkste Artefakt dieses Repositoriums, und die Methode ist reproduzierbar.

---

## Was entstanden ist

| | |
|---|---|
| Parallele Fragesätze | 100 |
| Sprachen | Persisch, Englisch, Deutsch |
| Trainingsbeispiele insgesamt | **300** |
| Aufteilung | 90 Training / 10 Validierung je Sprache |
| Verszeilen je Antwort | 4 bis 16, Mittelwert exakt 10,0 |
| Reimpaar Integrität | **300/300** Antworten haben eine gerade Zeilenzahl |
| System Prompts | 3, einer je Sprache, identisch über alle Beispiele |

Jede der 100 Fragen existiert in allen drei Sprachen mit einer inhaltlich passenden
Antwort. Die drei Sprachteile sind also echt parallel und nicht drei unabhängige
Datensätze.

---

## Warum ein KI erstellter Datensatz hier richtig war

Das Ziel ist eine anspruchsvolle literarische Form: Ferdowsis *Schahname* steht im
*Motaqāreb*-Versmaß, in Reimpaaren (*Masnawi*), mit einem klassischen Wortschatz, der sich
vom modernen Persisch deutlich unterscheidet. 100 solcher Passagen von Hand zu schreiben,
dann nochmals auf Englisch, dann auf Deutsch, inhaltlich deckungsgleich, bedeutet Wochen
an Facharbeit.

Drei Eigenschaften rechtfertigten den automatisierten Weg:

**1. Dreisprachige Parallelität ist fast kostenlos.** Dieselbe Antwort gleichzeitig in
drei Sprachen zu erzeugen und inhaltlich deckungsgleich zu halten, ist eine Aufgabe, bei
der ein Sprachmodell einen strukturellen Vorteil hat. Von Hand ist jede Sprache ein
eigener Übersetzungsdurchgang, mit Bedeutungsdrift bei jedem Schritt.

**2. Formale Konsistenz wird erzwungen, nicht erhofft.** Jede Antwort zielt auf dieselbe
Gestalt: Reimpaare, gerade Zeilenzahl, rund 10 Zeilen, gleiches Register. Das Ergebnis
zeigt sich in den Zahlen: 300 von 300 Antworten haben eine gerade Zeilenzahl, der
Mittelwert liegt bei exakt 10,0 Zeilen. Ein handgeschriebenes Korpus driftet.

**3. Hoher Reasoning Aufwand zahlt sich bei Versen aus.** Reimende, metrische Verse zu
schreiben ist keine Fließtext Aufgabe: Man muss ein Reimziel und eine Silbenzahl halten,
während man dichtet. Höherer Reasoning Aufwand verbessert das messbar, die Reimquoten
unten sind der Beleg.

---

## Gemessene Qualität des Ergebnisses

Reim lässt sich maschinell prüfen. Im *Masnawi* reimen die Zeilen paarweise (AA, BB, CC).
Entfernt man alles außer Buchstaben und vergleicht die Zeilenenden paarweise, ergibt sich
eine Zahl:

| Sprache | Reimende Verspaare | Gerade Zeilenzahl | Schriftreinheit |
|---|---|---|---|
| **Persisch** | **447/500 (89 %)** | 100/100 | keine fremden Zeichen |
| Deutsch | 363/500 (73 %) | 100/100 | entfällt |
| Englisch | 338/500 (68 %) | 100/100 | entfällt |

Persisch, die schwierigste der drei Sprachen und das eigentliche Ziel, schneidet am
besten ab. Das ist das Gegenteil dessen, was bei maschineller Übersetzung aus einem
englischen Original passiert, und zeigt, dass das Persische originär gedichtet und nicht
abgeleitet wurde.

Zum Vergleich: Die feinabgestimmten Modelle erreichen **33 bis 65 %** persische Reimquote.
**Der Datensatz ist deutlich besser als alles, was darauf trainiert wurde**, der Datensatz
ist also nicht der Engpass dieses Projekts.

---

## Die Pipeline und woher die Verlässlichkeit kommt

Geschwindigkeit nützt nur, wenn das Ergebnis vertrauenswürdig ist. Die Verlässlichkeit
entsteht hier dadurch, dass der Datensatz *in jeder Stufe prüfbar* ist, nicht dadurch,
dass man dem Generierungsschritt vertraut.

```
shahnameh_style_dataset.xlsx     ← Quelle der Wahrheit (von Hand editierbar)
         │  Glossar · Stilrichtlinie · Referenzverse · Quellen · Statistik
         ▼
   export_jsonl.py               ← Validierungsschranke
         │  Schema · Rollenreihenfolge · Leerfelder · Kodierung · Split-Integrität
         ▼
   training_data/*.jsonl         ← 8 Dateien, Training/Validierung je Sprache + gesamt
         │
         ▼
   Tokenizer-Round-Trip-Prüfung  ← vor dem Training gegen den echten Tokenizer
```

**Tabelle als Quelle der Wahrheit.** Der Datensatz liegt in einer Arbeitsmappe mit Glossar,
Stilrichtlinie, Referenzversen, Quellen und Statistikblatt: 369 Formeln, 0 Fehler. Sie
bleibt von Hand editierbar: Eine Persischlehrkraft kann eine Zeile korrigieren, ohne JSON
anzufassen.

**Validierung ist eine Schranke, kein Bericht.** `export_jsonl.py` verweigert die
JSONL Ausgabe, wenn die Rollenreihenfolge nicht stimmt, ein Feld leer ist oder ein Split
fehlerhaft ist. Fehlerhafte Daten können nicht unbemerkt ins Training gelangen.

**Round Trip Prüfung mit dem echten Tokenizer.** Vor jedem Trainingslauf wurden alle 300
Beispiele gegen den tatsächlichen Modell Tokenizer auf zweierlei geprüft: dass der Text
Kodierung und Dekodierung unverändert übersteht, und dass die Verlustmaskierung exakt die
Antwort Tokens abdeckt und sonst nichts. Alle 300 bestanden. Die längste Sequenz umfasst
635 Tokens bei einem Limit von 1024, nichts wird unbemerkt abgeschnitten.

Genau diese Prüfung erlaubte es später zu beweisen, dass ein Dekodierungsfehler *kein*
Datenproblem war: Maskierung und EOS Überwachung waren bereits verifiziert, der Fehler
musste also woanders liegen. **Frühe Verifikation ersparte später einen vergeblichen
Trainingsdurchlauf.**

---

## Eingesparte Zeit

| Aufgabe | Von Hand | Dieses Vorgehen |
|---|---|---|
| 100 persische Passagen im klassischen Versmaß | Wochen | eine Arbeitssitzung |
| Englische + deutsche Parallelfassungen | weitere Wochen | dieselbe Sitzung |
| Schema, Validierung, JSONL Export Werkzeug | 1 bis 2 Tage | Minuten |
| Arbeitsmappe mit Glossar, Stilrichtlinie, Statistik | 1 bis 2 Tage | Minuten |

Datensatz, Export Werkzeug, Validierung, Trainings Notebooks und Auswertungsgerüst
entstanden in wenigen Arbeitssitzungen. Die Konsistenz über 300 Beispiele in drei Sprachen
ist der Teil, der von Hand wirklich schwer zu erreichen ist, und genau der ist am
stärksten gelungen.

---

## Ehrliche Einschränkungen

Der Datensatz ist das beste Artefakt hier, und er trägt dennoch echte Vorbehalte:

- **Kein Muttersprachler hat die persischen Verse geprüft.** Reim und Schriftreinheit sind
  maschinell gemessen. **Versmaß (عروض), Grammatik und klassisches Register sind
  ungeprüft.** 89 % Reimquote besagt, dass die Form weitgehend stimmt; sie beweist nicht,
  dass die Verse metrisch aufgehen.
- **Die Verse sind neu im Stil Ferdowsis verfasst. Es ist nicht sein Text.** Keine Zeile
  ist ein Zitat aus dem *Schahname*.
- **100 Fragen sind wenig.** Genug, um eine Stimme zu vermitteln; nicht genug für Prosodie.
  Was das beheben würde, steht in der Haupt README.
- **Englisch und Deutsch reimen weniger konsistent** (68 % und 73 %) als Persisch, teils
  weil der Reim in diesen Traditionen weniger strukturbildend ist, teils weil Persisch
  Priorität hatte.

---

## Reproduktion

Die Arbeitsmappe ist die Eingabe; alles Nachgelagerte wird neu erzeugt:

```bash
python export_jsonl.py      # xlsx -> training_data/*.jsonl, mit Validierung
python make_visuals.py      # alle Diagramme aus den Trainingsprotokollen neu erzeugen
```

Tabelle bearbeiten, neu exportieren, neu trainieren. Die Verse lassen sich von einer
Fachperson korrigieren, ohne Code anzufassen, genau der Arbeitsablauf, den eine
muttersprachliche Prüfung des Persischen benötigen würde.
