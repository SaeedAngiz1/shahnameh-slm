# Unsloth und LoRA: wie das Training tatsächlich funktioniert

**Sprache:** [English](unsloth-lora.md) · [فارسی](unsloth-lora.fa.md) · Deutsch

---

Dieses Projekt trainiert ein Modell mit 4 Milliarden Parametern auf einer 16 GB GPU der
Consumer Klasse, kostenlos. Möglich wird das durch zwei kombinierte Techniken: **LoRA**
(nur einen winzigen Bruchteil der Gewichte trainieren) und **4 Bit Quantisierung** (den
Rest komprimiert speichern), zusammengehalten von **Unsloth** (schnell machen und dafür
sorgen, dass beides korrekt zusammenspielt).

---

## Das Problem

Qwen3-4B hat rund 4 Milliarden Parameter. Klassisches vollständiges Fine Tuning muss je
Parameter vorhalten:

| Was | Genauigkeit | Speicher für 4B |
|---|---|---|
| Modellgewichte | fp16 | ~8 GB |
| Gradienten | fp16 | ~8 GB |
| Adam Optimiererzustand (2 Momente) | fp32 | ~32 GB |
| **Summe vor Aktivierungen** | | **~48 GB** |

Eine Tesla T4 hat **16 GB**. Vollständiges Fine Tuning passt nicht annähernd. Es fehlt
der Faktor drei, noch vor den Aktivierungen.

---

## LoRA: eine kleine Korrektur statt des ganzen Modells trainieren

**Low-Rank Adaptation** friert die ursprünglichen Gewichte vollständig ein und lernt zu
jeder angesprochenen Gewichtsmatrix eine kleine additive Korrektur.

Statt für eine Gewichtsmatrix `W` der Form `d × k` ein gleich großes Update `ΔW` zu
lernen, lernt LoRA zwei schmale Matrizen:

```
W_effektiv = W + (α / r) · B · A

    A :  r × k      (r ist der „Rang", hier 16 oder 64)
    B :  d × r
    W :  eingefroren, wird nie aktualisiert
```

Da `r` klein ist, hat `B · A` weit weniger Parameter als `W`, erzeugt aber dennoch ein
Update der Form `d × k`. Nur `A` und `B` erhalten Gradienten. Der Optimiererzustand
schrumpft entsprechend, dort liegt die eigentliche Ersparnis.

### Die hier verwendeten Einstellungen

Aus `adapter_config.json`, strukturell identisch in beiden Läufen:

```json
{
  "r": 64,                    // 16 im mehrsprachigen Lauf
  "lora_alpha": 64,           // stets gleich r
  "lora_dropout": 0,
  "bias": "none",
  "use_rslora": false,
  "task_type": "CAUSAL_LM",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
}
```

**`lora_alpha` ist bewusst gleich `r`.** Der Skalierungsfaktor ist `α / r`; setzt man
beide gleich, bleibt er bei jedem Rang exakt 1,0. Eine Rangerhöhung von 16 auf 64 fügt
dann Kapazität hinzu, *ohne* zugleich die Größe des Updates zu vervielfachen: es ändert
sich eine Variable, nicht zwei. Bliebe `alpha` fest, während `r` sich ändert, wäre eine
Rangänderung stillschweigend auch eine Lernratenänderung, und der Vergleich der beiden
Läufe wäre wertlos.

**Alle sieben Projektionsmatrizen werden angesprochen**, nicht nur die Attention. Nur
`q_proj`/`v_proj` anzusprechen ist eine verbreitete, günstigere Voreinstellung, aber
Stiltransfer verändert, *wie* das Modell schreibt, und das sitzt wesentlich in den
MLP Blöcken (`gate_proj`, `up_proj`, `down_proj`). Sie auszulassen hieße, denselben Effekt
mit weniger Netz erzwingen zu wollen.

**`lora_dropout = 0`.** Dropout schützt vor Überanpassung. Dieses Projekt *will* die Form
hart anpassen (siehe die Überanpassungs Diskussion in der Haupt README), daher ist es
abgeschaltet.

### Was der Rang kostet

| Rang | Adaptergröße | Persische Reimquote | Nebenwirkung |
|---|---|---|---|
| r=16 | 132 MB | 33 % | keine, saubere Schrift |
| r=64 | 520 MB | **65 %** | fremde Zeichen dringen ein |

Der höhere Rang erfasste die poetische Form deutlich besser. Zusammen mit 12 Epochen
verschlechterte er das Modell aber so weit, dass chinesische und kyrillische Zeichen in
persischen Versen auftauchten. **Kapazität ist nicht umsonst.**

---

## 4 Bit Quantisierung: den eingefrorenen Teil verkleinern

LoRA beseitigt die Kosten für Optimierer und Gradienten, doch die eingefrorenen
Basisgewichte müssen weiterhin im Speicher liegen: rund 8 GB in fp16, also fast eine
ganze T4, noch vor allen Aktivierungen.

Das Basismodell wird daher in **4 Bit** geladen (`load_in_4bit=True`), über das
vorquantisierte `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. Die Gewichte sinken auf
etwa 2,5 GB. Sie werden im Vorwärtsdurchlauf zur Laufzeit dequantisiert; da sie
eingefroren sind und nie aktualisiert werden, summiert sich der Quantisierungsfehler über
das Training nicht auf.

LoRA über einer eingefrorenen 4 Bit Basis ist das, was der Name **QLoRA** bezeichnet.

Den vorquantisierten Checkpoint zu verwenden statt beim Laden zu quantisieren, spart
zudem einen Schritt, der sonst bei jedem Trainingsstart erneut anfiele.

### Das resultierende Budget auf einer T4

| | Speicher |
|---|---|
| Basisgewichte, 4 Bit | ~2,5 GB |
| LoRA Parameter + Gradienten + Optimierer | ~0,3 GB |
| Aktivierungen (mit Gradient Checkpointing, Seq 1024, Batch 2) | ~2 bis 4 GB |
| **Summe** | **bequem innerhalb von 16 GB** |

---

## Unsloth: was es beiträgt

[Unsloth](https://github.com/unslothai/unsloth) ist die Schicht, die das Obige schnell und
korrekt macht, ohne handgeschriebenes CUDA.

**Fusionierte Triton Kernel.** Handgeschriebene Kernel für Attention, RoPE, RMSNorm, den
MLP Block und Cross Entropy ersetzen den Standard PyTorch Pfad. Etwa doppelt so schnelles
Training bei geringerem Speicherbedarf, praktisch heißt das hier: ein 12 Epochen Lauf in
**724 Sekunden**.

**Manuelles Autograd für den LoRA Pfad.** Statt sich auf den generischen Autograd Graphen
zu verlassen, implementiert Unsloth Rückwärtsdurchläufe, die große Zwischenergebnisse gar
nicht erst materialisieren, daher der Großteil der Speicherersparnis.

**Architekturgerechtes Patchen.** Qwen3, Gemma und Llama brauchen jeweils andere
Behandlung. Unsloth erkennt die Architektur und patcht entsprechend; das Startbanner meldet,
was geschehen ist:

```
==((====))==  Unsloth 2026.9.4: Fast Qwen3 patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.562 GB.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
```

Die Zeile `Bfloat16 = FALSE` macht die T4 Einschränkung sichtbar. GPUs der Turing Generation
unterstützen kein bf16, alles läuft also in fp16, genau deshalb fiel die Wahl in diesem
Projekt auf Qwen3 statt Gemma, dessen Familie in fp16 anfälliger für NaN Verluste ist.

**Bereitstellung vorquantisierter Modelle.** Unsloth veröffentlicht `-unsloth-bnb-4bit`-
Varianten und erspart damit einen Quantisierungsdurchlauf bei jedem Laden.

### Die API

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,                    # None = passend zur GPU wählen (fp16 auf T4)
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64, lora_alpha=64, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

Danach läuft das Training mit einem gewöhnlichen HuggingFace-`Trainer`/`SFTTrainer`.
Unsloth ersetzt das Innenleben, nicht die Schnittstelle.

**Ein Detail für die Inferenz:**

```python
FastLanguageModel.for_inference(model)   # ~2x schnellere Generierung
```

Das wegzulassen ist kein Fehler, nur unnötig langsam. Manche Versionen stellen es nicht
bereit, daher rufen die Skripte hier es defensiv auf.

---

## Warum LoRA gerade für dieses Projekt passt

Über den Speicher hinaus passt LoRA zur Aufgabe:

**Der Datensatz ist winzig.** 90 bis 270 Beispiele. 4 Milliarden Parameter auf 90 Beispielen
zu aktualisieren würde allgemeine Sprachfähigkeit durch Rauschen überschreiben. Das Update
auf eine niedrigrangige Korrektur zu beschränken, wirkt selbst als Regularisierung.

**Das Ziel ist Stil, nicht Wissen.** Das Modell kann bereits Persisch. Ihm muss beigebracht
werden, *wie* es schreiben soll: Reimpaare in klassischem Register. Das ist eine Änderung
der Ausgabeverteilung, keine Einspeisung von Fakten, und genau darin ist LoRA am besten.

**Adapter sind klein und austauschbar.** 132 MB und 520 MB gegenüber 8 GB für ein
vollständiges Modell. Beide Adapter teilen sich ein Basismodell; sie zu vergleichen heißt,
eine kleine Datei zu tauschen, statt zwei komplette Modelle vorzuhalten.

**Das Basismodell bleibt unversehrt.** Die eingefrorenen Gewichte können nicht beschädigt
werden. Das Schlimmste, was ein misslungener Lauf hinterlässt, ist ein schlechter Adapter,
und der wird gelöscht.

### Wo LoRA nicht gerettet hat

LoRA beschränkt, *welche* Gewichte sich ändern, nicht *wie weit* sie driften. Lauf 2
(Rang 64, 12 Epochen) überanpasste dennoch so stark, dass die Schriftkonsistenz brach.
**Ein kleiner Adapter macht Überanpassung nicht unmöglich.** Rang, Epochen und Lernrate
brauchen dieselbe Sorgfalt wie beim vollständigen Fine Tuning.

---

## Vergleich: das vollständige Fine Tuning in diesem Projekt

Das Modell SmolLM2-360M wurde bewusst als Kontrast **vollständig** feinabgestimmt:

| | SmolLM2-360M | Qwen3-4B |
|---|---|---|
| Methode | vollständiges Fine Tuning | LoRA + 4 Bit |
| Trainierbare Parameter | alle 360M | ~1 bis 2 % von 4B |
| Genauigkeit | fp16 gemischt | 4 Bit Basis, fp16 Rechnung |
| Ergebnisartefakt | 1,3 GB Modell | 132/520 MB Adapter |
| GPU Nutzung | 2× T4 via DDP | eine einzelne T4 |
| Lernrate | 2e-5 | 2e-4 / 3e-4 |

Die Lernraten unterscheiden sich um **eine Größenordnung**. Das ist erwartet und kein
Fehler: LoRA aktualisiert eine kleine niedrigrangige Korrektur und verträgt, und braucht,
eine weit höhere Lernrate als vollständiges Fine Tuning, wo 2e-4 über alle Gewichte
zerstörerisch wäre.

Bei 360M passt vollständiges Fine Tuning und ist einfach. Bei 4B passt es auf dieser
Hardware überhaupt nicht. Ungefähr dort hört LoRA auf, eine Bequemlichkeit zu sein, und
wird zur einzigen Option.

---

## Das Ganze auf Kaggle ausführen

Diese Seite behandelt die Methode. Die praktische Seite (Einrichtung von Beschleuniger und
Internet, die fixierten Installationen und warum sie je Modellfamilie
abweichen, beide T4 ansteuern, Läufe über die Kaggle API automatisieren sowie die
Fallen bei Pfaden und Kodierung, die echte Zeit gekostet haben) steht in
**[Unsloth auf Kaggle betreiben](kaggle-unsloth.de.md)**.

---

## Weiterführendes

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): Dettmers et al., 2023
- [Unsloth Dokumentation](https://docs.unsloth.ai/)
