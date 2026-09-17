# Shahnameh SLM

**Sprache:** [English](README.md) · [فارسی](README.fa.md) · Deutsch

Kleinen Sprachmodellen beibringen, jede Frage in der Stimme von Ferdowsis *Schahname* zu
beantworten, auf **Persisch, Englisch und Deutsch**, auf zwei kostenlosen Kaggle T4 GPUs.

<p align="center">
  <img src="docs/assets/06-progress.gif" width="640" alt="Persische Reimtreue über die Trainingsläufe">
</p>

Dieses Repositorium enthält den Datensatz, den Trainingscode, drei trainierte Modelle und,
ungewöhnlicherweise, eine ehrliche Aufzeichnung dessen, was alles schiefging. Die
Fehlschläge erwiesen sich als der nützlichste Teil und sind daher ebenso sorgfältig
dokumentiert wie die Erfolge.

**Wo es die Modelle gibt**

Alle drei sind auf dem Hugging Face Hub veröffentlicht, unter
[huggingface.co/SaeedAngiz1](https://huggingface.co/SaeedAngiz1):
[`shahnameh-qwen3-4b-fa-lora`](https://huggingface.co/SaeedAngiz1/shahnameh-qwen3-4b-fa-lora),
[`shahnameh-qwen3-4b-multilingual-lora`](https://huggingface.co/SaeedAngiz1/shahnameh-qwen3-4b-multilingual-lora) und
[`shahnameh-smollm2-360m`](https://huggingface.co/SaeedAngiz1/shahnameh-smollm2-360m).
Wie man sie lädt, steht unter [Modelle ausführen](#modelle-ausführen).

**Weiterführende Dokumentation**
- [Wie der Datensatz entstand](docs/dataset-creation.de.md): erstellt mit Claude Code (Opus 5, besonders hoher Reasoning Aufwand)
- [Unsloth und LoRA im Detail](docs/unsloth-lora.de.md): wie 4 Milliarden Parameter auf 16 GB passen
- [Unsloth auf Kaggle betreiben](docs/kaggle-unsloth.de.md): die Praxisanleitung für Einrichtung, Installation, beide GPUs, die API und die Stolperfallen

---

## Was es tut

Stellen Sie eine Frage in einer der drei Sprachen; Sie erhalten eine Antwort in
Schahname Versen.

```
Q: رستم که بود؟

شه ترک، ای خردمند و پاکدست
که توران بدو بود، چون بیشه بکست
سپهبد بود، ای برادر مهتر
به پیش از خوانه، هیچ کس نهاده بر
چو رستم بکشت پیشانی ز هر مهر
بیامد در رخش، یکی شاخ ز گهر
تهمتن بکشت او را ز خاک زرد
به گیتی نان خواست، یکی شاخ زرد
```
*80 % der Verspaare reimen korrekt · keine Schriftverunreinigung · `تهمتن` und `رخش` sind
authentischer Epenwortschatz*

```
Q: Ist es besser zu reden oder zu schweigen?

Zu viel Rede ist die Quelle des Leids,
Wenig Wort wie eine Blume glänzt.
Doch wenn du sprichst, so sprich mit klarem Sinn;
Die Wahrheit ist der Edelste, der reine Sinn.
```

Das Ziel ist **Stil, nicht Stoffkenntnis**. Bewertet wird das Modell an Versmaß, Reim und
klassischem Register, nicht daran, ob es die Genealogie des Schahname richtig wiedergibt.
(Das tut es häufig nicht. Siehe [Einschränkungen](#einschränkungen).)

---

## Ergebnisse

<p align="center">
  <img src="docs/assets/02-rhyme-rate.png" width="760" alt="Reimquote nach Lauf">
</p>

Persische Verse in der *Masnawi*-Form reimen paarweise: AA, BB, CC. Dadurch wird Qualität
teilweise **messbar**: alles außer persischen Buchstaben entfernen, die Enden jedes
Zeilenpaars vergleichen, zählen. Der Datensatz erreicht 89 %. Diese Zahl ist das Ziel, und
der Abstand dorthin ist der eigentliche Fortschrittsmaßstab des Projekts.

| Lauf | Konfiguration | Persische Reimquote | Schriftverunreinigung |
|---|---|---|---|
| Trainingsdaten | Ziel | **89 %** | keine |
| Lauf 2 | Qwen3-4B, nur Persisch, 12 Epochen, LoRA r=64 | **65 %** | 6 Zeichen auf 10 Antworten |
| Lauf 1 | Qwen3-4B, 3 Sprachen, 3 Epochen, LoRA r=16 | 33 % | keine |
| SmolLM2 | 360M vollständiges Fine Tuning | entfällt (Ausgabe war entartet) | keine |

Jede Zahl stammt aus einem Trainingsprotokoll oder einer gemessenen Auswertung in diesem
Repositorium. `make_visuals.py` erzeugt jedes Diagramm aus diesen Dateien neu; nichts ist
illustrativ.

---

## Warum zwei kostenlose T4 die richtige Wahl waren

Das gesamte Projekt läuft auf **Kaggles kostenloser GPU Stufe**: 2× NVIDIA Tesla T4, je
16 GB, 30 Stunden pro Woche. Die gesamte aufgewendete Rechenzeit aller hier dokumentierten
Läufe liegt unter drei Stunden.

**Was das ermöglicht.** Ein Modell mit 4 Milliarden Parametern passt mit 4 Bit QLoRA
bequem auf eine T4. Der reine Persisch Lauf dauerte **724 Sekunden**, der mehrsprachige
**411 Sekunden**. Das ist für diese Art Arbeit kein Kompromiss, sondern schlicht
ausreichend.

**Wie die zweite GPU genutzt wird.** Statt einen Job auf beide Karten aufzuteilen, laufen
standardmäßig *zwei verschiedene Jobs parallel*: GPU 0 trainiert, während GPU 1
Basisantworten des untrainierten Modells erzeugt. Diese Basisantworten braucht man ohnehin,
um zu belegen, dass das Fine Tuning etwas verändert hat, sonst wäre es Leerlaufzeit.
Verteiltes Training (`GPU_MODE = "ddp"`) ist verfügbar, aber bewusst optional.

**Der ehrliche Vorbehalt zur Verlässlichkeit.** Die T4 selbst fielen nie aus. Aber das
*verteilte* Training auf zwei GPUs schon. Eine NCCL Kollektivabweichung zerstörte den
Export Schritt des ersten Laufs, nachdem das Training bereits erfolgreich beendet war. Die
Lösung war architektonisch, kein Neuversuch: Training und Export in getrennte Prozesse
trennen. Kostenlose Hardware ist verlässlich; die verteilte Koordination ist der fragile
Teil.

**T4 spezifische Randbedingungen, die den Code prägen.** Die T4 gehört zur
Turing Generation: **kein bfloat16**, kein FlashAttention 2. Alles läuft daher in
fp16 Mixed Precision, und die Modellauswahl wird danach gefiltert, ob sie in fp16 auf
Turing funktioniert. Genau diese eine Randbedingung führte zur Wahl von Qwen3 statt Gemma.

---

## Entwurfsentscheidungen und ihre Gründe

### Warum Qwen3-4B-Instruct

Erster ernsthafter Kandidat war Gemma 4 E4B (140 Sprachen, hervorragende mehrsprachige
Abdeckung). Er wurde vor dem ersten Lauf verworfen, aus zwei Gründen: Es verlangt
`transformers 5.10.1`, einen auf diesem Konto ungetesteten Stack, und Modelle der
Gemma Familie neigen am ehesten zu NaN Verlusten in fp16, genau der Genauigkeit, zu der
eine T4 zwingt. Qwen3-4B verlangt `transformers 4.56.2`, das hier bereits erfolgreich
installiert und trainiert hatte, und verfügt über starkes persisches Vortraining.

Diese Entscheidung hielt: Qwen3 trainierte auf Anhieb, ohne NaN.

Es brachte zugleich den interessantesten Fehler des Projekts hervor. Qwen3 ist chinesischer
Herkunft, und das zeigt sich unter Belastung.

### Warum LoRA statt vollständigem Fine Tuning

Das 360M Modell wurde vollständig feinabgestimmt; die 4B Modelle nutzen LoRA. Bei 4B passt
vollständiges Fine Tuning nicht in 16 GB, und bei nur 90 bis 270 Trainingsbeispielen wäre es
ohnehin das falsche Werkzeug. `lora_alpha` ist gleich `r` gesetzt, sodass der
LoRA Skalierungsfaktor unabhängig vom Rang bei 1,0 bleibt.

**Ausführlich: [Unsloth und LoRA](docs/unsloth-lora.de.md).**

### Warum Verlust nur auf der Antwort

Das Training maskiert den Prompt: `labels = [-100] * len(prompt) + answer + [eos]`. Das
Modell wird nur an den Versen bewertet, die es erzeugt, nie am Wiederholen der Frage.

Entscheidend: **Das EOS Token steht in den Labels.** Das ist der häufigste Fine Tuning-
Fehler: lässt man es weg, lernt das Modell nie aufzuhören. Dass es vorhanden war, bewies
später, dass das SmolLM2 Wiederholungsdesaster *kein* Trainingsfehler war.

### Warum ein eigenes Chat Template

SmolLM2 bringt kein Chat Template mit, daher wird das Trainingsformat explizit definiert
und als `chat_template.jinja` im Tokenizer gespeichert, sodass `apply_chat_template()` bei
der Inferenz das Trainingsformat zeichengenau reproduziert. Eine Abweichung hier ist
lautlos und verheerend.

### Warum der Validierungssatz nie trainiert wird

30 Beispiele (10 je Sprache) sind zurückgehalten. Jede Zahl in dieser README ist daran
gemessen.

---

## Die drei Modelle

| Modell | Basis | Methode | Größe | Stärke |
|---|---|---|---|---|
| [`shahnameh-qwen3-4b-fa-lora`](https://huggingface.co/SaeedAngiz1/shahnameh-qwen3-4b-fa-lora) | Qwen3-4B-Instruct-2507 | LoRA r=64, 12 Epochen, nur Persisch | 520 MB | **Persische Verse**: 65 %, Spitze 80 % |
| [`shahnameh-qwen3-4b-multilingual-lora`](https://huggingface.co/SaeedAngiz1/shahnameh-qwen3-4b-multilingual-lora) | Qwen3-4B-Instruct-2507 | LoRA r=16, 3 Epochen, fa/en/de | 142 MB | **Konsistenz**: saubere Schrift in allen 3 Sprachen |
| [`shahnameh-smollm2-360m`](https://huggingface.co/SaeedAngiz1/shahnameh-smollm2-360m) | SmolLM2-360M | vollständiges Fine Tuning | 1,3 GB | **nur Englisch**; läuft auf CPU |

Alle drei liegen auf dem Hugging Face Hub unter [SaeedAngiz1](https://huggingface.co/SaeedAngiz1); die Modellkarten tragen dieselben Messwerte wie diese README.

### Zur Konsistenz

Die beiden Qwen3 Modelle stehen im Zielkonflikt, und genau das ist der Punkt.

**Lauf 2 schreibt besseres Persisch, ist aber instabiler.** Über zehn zurückgehaltene
Fragen erreicht er im Mittel 65 % Reimquote, einzelne Antworten schwanken jedoch zwischen
**80 % und 60 %**, und nur dieses Modell lässt nicht persische Zeichen durch. Lauf 1 tut
das nie, bleibt aber bei 33 %.

Die Ausgabe schwankt zudem **zwischen mehreren Läufen derselben Frage** bei
`temperature=0.7`. Eine einzelne gute Probe ist kein Beleg für Qualität, eine einzelne
schlechte kein Beweis für Untauglichkeit. Alle Kennzahlen hier sind Mittelwerte über den
gesamten Validierungssatz.

Die allgemeine Regel, auf die dieses Projekt immer wieder stieß: **Die Einstellungen, die
die poetische Form stark genug einprägen, sind dieselben, die das Modell zu beschädigen
beginnen.**

---

## Aufgetretene Probleme

### 1. Verteiltes Training starb, nachdem das Training gelungen war

Der SmolLM2 Lauf absolvierte alle 51 Schritte auf beiden T4 und stürzte dann beim Laden
des besten Checkpoints für den Export an abweichenden NCCL ALLGATHER Sequenznummern ab.
Kaggle bewahrte die Checkpoints, daher bestand die Lösung in einem separaten
Einzelprozess Notebook für Auswertung und Export. **Lehre: Export gehört nicht in denselben
Prozess wie verteiltes Training.**

### 2. Greedy Decoding ließ ein funktionierendes Modell wertlos erscheinen

<p align="center">
  <img src="docs/assets/04-termination.png" width="760" alt="Terminierung mit und ohne Wiederholungsstrafe">
</p>

Die SmolLM2 Proben wiederholten eine Zeile bis zum Token Limit, in allen drei Sprachen. Der
naheliegende Schluss (schlechter Checkpoint, neu trainieren) war falsch.

Führt man die Referenzantworten erzwungen durch das Modell und liest die Verteilung an der
Stelle, an der EOS stehen müsste, vergibt das Modell **P(EOS) = 0,55**, und `<|endoftext|>`
ist in **17 von 30 Fällen die Top 1 Vorhersage**. Es wusste genau, wo es aufhören muss.
Greedy Decoding geriet lediglich in eine selbstverstärkende Schleife.

`repetition_penalty=1.15` brachte die Terminierung von **0/30 auf 25/30**. Ohne
Neutraining. **Lehre: Bevor man den Checkpoint beschuldigt, messen, ob er die richtige
Antwort kennt.**

### 3. Die Lösung des einen Modells war der Fehler des nächsten

<p align="center">
  <img src="docs/assets/03-decoding-sweep.png" width="760" alt="Decoding Vergleich gegen persische Reimquote">
</p>

Dieselbe `repetition_penalty` auf Qwen3 übertragen **schadete** den persischen Versen.
Masnawi Reim lebt von wiederkehrenden Lauten und formelhaften Funktionswörtern (`به`, `ز`,
`که`, `چو`), und Wiederholung zu bestrafen drängt das Modell von genau dem weg, was die Form
verlangt. Ohne die Strafe stieg die Reimquote von 39 % auf 53 %.

**Lehre: Decoding Parameter sind modellspezifisch und gehören mit der Messung daneben ins
Preset.**

### 4. Härteres Training brach das Modell auf neue Weise

<p align="center">
  <img src="docs/assets/01-overfitting.png" width="760" alt="Auseinanderlaufender Trainingsverlust und Validierungsverlust">
</p>

Lauf 2 ging hart vor: 12 Epochen, Rang 64, höhere Lernrate. Der Validierungsverlust
erreichte sein Minimum bei **Epoche 3 (2,43)** und stieg bis Epoche 12 auf **3,71**,
während der Trainingsverlust auf **0,84** fiel: Auswendiglernen im Lehrbuchsinn.

Die Reimquote verbesserte sich dennoch (33 % → 65 %), weshalb `load_best_model_at_end`
bewusst auf `False` gesetzt wurde: Für Stiltransfer ist die Epoche mit dem niedrigsten
Validierungsverlust nicht das beste Modell.

Der Preis zeigte sich dort, wo der Verlust nichts misst. Lauf 2 begann, Zeichen **anderer
Schriftsysteme** in persische Verse einzustreuen:

> `که لغزش، دل را کند تیره‌眼`

`眼` ist chinesisch für „Auge". Das Modell griff nach einem Wort für Sehen und erwischte
das chinesische Token statt des persischen `چشم`. Qwen3 ist chinesischer Herkunft, diese
Token tragen also durchweg hohe A priori Wahrscheinlichkeit; die Überanpassung verschlechterte
die Ausgabeverteilung so weit, dass sie nicht mehr zuverlässig unterdrückt wurden. Lateinische
(`az`) und kyrillische (`ен`) Zeichen drangen genauso ein. Lauf 1 hatte davon kein einziges.

**Lehre: Der Validierungsverlust erfasst nicht jede Fehlerart. Messen Sie das, worauf es
ankommt**: hier Schriftreinheit und Reim, die der Verlust beide nicht sieht.

---

## Einschränkungen

- **Fakten sind häufig falsch.** Das englische Modell hat Rostam schon zum Gefolgsmann
  „König Tamerlans" gemacht (14. Jahrhundert, rund 400 Jahre nach Ferdowsi), den Namen
  seines Pferdes erfunden und Sohrab zu seinem Bruder statt seinem Sohn gemacht. 90 bis 270
  Beispiele lehren eine Stimme, keine Genealogie.
- **Das Persische ist nicht als *gutes* Persisch verifiziert.** Reim und Schriftreinheit
  sind maschinell gemessen. **Versmaß (عروض), Grammatik und klassisches Register wurden nie
  von einem Muttersprachler geprüft.**
- **Erfundene Wörter bleiben.** Nichtwörter wie `ستوس`, `خامور` und `بهاز` treten bei allen
  bisher erprobten Einstellungen auf.
- **Deutsch ist flüssig, aber thematisch abwegig.**
- **Die Qualität schwankt von Lauf zu Lauf.** Nach Mittelwerten urteilen, nicht nach
  Lieblingsproben.

### Was das wirklich beheben würde

Der Datensatz ist sauber, aber klein: **90 persische Beispiele können keine klassische
Prosodie vermitteln.** Jede bisherige Hyperparameteränderung hat einen Mangel gegen einen
anderen getauscht.

Die eigentliche Lösung ist authentischer Text. Ferdowsis Schahname ist gemeinfrei, rund
50.000 echte Verspaare. Das übliche zweistufige Rezept:

1. Fortgesetztes Training auf echten Verspaaren: lernt Versmaß und Diktion von Ferdowsi selbst
2. Fine Tuning auf den Frage Antwort Paaren: lernt darauf das Antwortformat

Derzeit soll ein kleiner Datensatz beides zugleich leisten: *was* zu sagen ist und *wie*.
Beides zusammen kann er nicht.

---

## Modelle ausführen

### Von Hugging Face herunterladen

Alle Modelle liegen auf dem Hugging Face Hub unter
[SaeedAngiz1](https://huggingface.co/SaeedAngiz1). SmolLM2 ist ein vollständiger Satz
Gewichte und lädt für sich allein. Die beiden Qwen3 Modelle sind LoRA Adapter und laden auf
`Qwen/Qwen3-4B-Instruct-2507` auf, das der Hub mitholt.

```python
# SmolLM2, das vollständige Modell, CPU genügt
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "SaeedAngiz1/shahnameh-smollm2-360m"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

```python
# Qwen3 Adapter auf dem Basismodell, GPU nötig
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", device_map="auto")
model = PeftModel.from_pretrained(base, "SaeedAngiz1/shahnameh-qwen3-4b-fa-lora")
```

Nur die Dateien holen, und SmolLM2 gleich dorthin legen, wo `ask_smollm2.py` es erwartet:

```bash
hf download SaeedAngiz1/shahnameh-smollm2-360m --local-dir smollm2-shahnameh-model
```

### Lokal (SmolLM2, CPU, keine GPU nötig)

```bash
python ask_smollm2.py "Who was Rostam?"
python ask_smollm2.py --lang fa "رستم که بود؟"
python ask_smollm2.py --temperature 0.8 "What is wisdom?"
```

### Auf Kaggle (Qwen3, das gute Persisch)

Das Notebook [`shahnameh-ask`](https://www.kaggle.com/code/saeed010/shahnameh-ask) bietet
ein Eingabefeld: Zellen ausführen, Frage eintippen, und die Antwort erscheint zusammen mit
ihrer Reimquote und etwaigen nicht persischen Zeichen.

### Training reproduzieren

`kaggle/shahnameh_unsloth_kaggle.ipynb` ist die maßgebliche Quelle. Die 8 JSONL Dateien aus
`training_data/` als Kaggle Datensatz hochladen, anhängen, Beschleuniger auf **GPU T4 x2**
mit Internet stellen, „Run All".

---

## Danksagungen

- **[Unsloth](https://github.com/unslothai/unsloth)**: 2× schnelleres LoRA Fine Tuning
- **[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** (Alibaba): Apache 2.0
- **[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)** (Hugging Face): Apache 2.0
- **[Kaggle](https://www.kaggle.com/)**: kostenlose 2× T4 GPU Rechenzeit
- **Claude Code (Opus 5)**: [Erstellung und Validierung des Datensatzes](docs/dataset-creation.de.md)
- **Abu'l-Qāsim Ferdowsi**: das *Schahname*, vollendet 1010 n. Chr.

Die Trainingsverse sind neu in Ferdowsis Stil verfasst; es ist nicht sein Text.

## Lizenz

Code und Datensatz: **Apache 2.0**. Die Modell Adapter übernehmen die Apache 2.0 Lizenzen
ihrer Basismodelle.
