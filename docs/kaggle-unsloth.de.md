# Unsloth auf Kaggles kostenloser Zwei GPU Stufe betreiben

**Sprache:** [English](kaggle-unsloth.md) · [فارسی](kaggle-unsloth.fa.md) · Deutsch

---

Dies ist der praktische Begleiter zu [Unsloth und LoRA](unsloth-lora.de.md). Jene Seite
erklärt, *warum* die Methode in 16 GB passt; diese erklärt, *wie man sie tatsächlich
ausführt*: Einrichtung, Dateisystem, Installation, beide GPUs ansteuern, Automatisierung
vom Terminal aus, und was konkret schiefging.

Alles hier lief auf der kostenlosen Stufe. In diesem Projekt wurde nirgends bezahlte
Rechenzeit verwendet.

---

## Was die kostenlose Stufe tatsächlich bietet

| Ressource | Kontingent |
|---|---|
| GPU | **2× NVIDIA Tesla T4**, je 16 GB |
| GPU Kontingent | **30 Stunden pro Woche**, wöchentlich zurückgesetzt |
| Maximale Sitzungsdauer | 9 Stunden (GPU), 12 Stunden (CPU/TPU) |
| TPU Kontingent | 20 Stunden pro Woche |
| Ausgabespeicher (`/kaggle/working`) | 20 GB, als Notebook Ausgabe gespeichert |
| Internetzugang | verfügbar, erfordert aber ein **telefonisch verifiziertes Konto** |

Das gesamte Projekt (drei Trainingsläufe, ein Decoding Vergleich, ein
Auswertungs Notebook) verbrauchte **weniger als drei Stunden** dieses
30 Stunden Wochenbudgets.

Zwei Randbedingungen wiegen schwerer als alle anderen:

**Internet erfordert Telefonverifizierung.** Ohne sie schlägt `pip install unsloth` fehl
und das Modell kann nicht von Hugging Face geladen werden. Das ist der häufigste Grund,
warum ein Notebook, das „eigentlich funktionieren müsste", es nicht tut.

**Die T4 gehört zur Turing Generation: kein bfloat16.** Alles läuft in fp16. Das ist keine
Nebensächlichkeit. Es schließt ganze Modellfamilien aus, die in fp16 zu NaN Verlusten
neigen, und ist der Grund, warum dieses Projekt Qwen3 statt Gemma gewählt hat.

---

## Ein Notebook einrichten

Im rechten Bedienfeld des Kaggle Editors:

1. **Session options → Accelerator → GPU T4 x2.** Nicht „GPU P100", das ist eine einzelne Karte.
2. **Session options → Internet → On.**
3. **Add Input → Datasets** → Trainingsdaten anhängen.

Dann *Run All*. Alle einstellbaren Parameter dieses Projekts stehen in einer einzigen
Zelle ganz oben.

### Das Dateisystem

| Pfad | Zweck |
|---|---|
| `/kaggle/input/<dataset-slug>/` | angehängte Datensätze, **nur lesbar** |
| `/kaggle/working/` | Ihre Ausgabe, **bleibt erhalten**, wenn das Notebook gespeichert wird |
| `/kaggle/temp/` | Zwischenspeicher, wird am Sitzungsende verworfen |

Datensätze werden in unvorhersehbarer Tiefe eingehängt, daher fixiert dieses Projekt nie
einen Pfad, sondern sucht nach einem bekannten Dateinamen:

```python
def find_data_dir(root):
    """Kaggle hängt Datensätze in unbekannter Tiefe ein; Ordner über bekannte Datei finden."""
    matches = sorted(Path(root).rglob("all_train.jsonl"))
    if not matches:
        raise FileNotFoundError(f"kein all_train.jsonl unter {root}; Datensatz angehängt?")
    return matches[0].parent
```

Diese Gewohnheit hat sich mehrfach ausgezahlt. Siehe [Stolperfallen](#stolperfallen-die-echte-zeit-gekostet-haben).

---

## Unsloth auf Kaggle installieren

Die Kaggle Images bringen eigene Versionen von PyTorch und Transformers mit, die häufig mit
dem kollidieren, was Unsloth benötigt. Die Installationen sind daher **fixiert und je
Modellfamilie verschieden**:

```python
if MODEL_PRESET == "gemma-4-e4b":
    !pip install -q unsloth
    !pip install -q --no-deps transformers==5.10.1 "tokenizers>=0.22.0,<=0.23.0"
    !pip install -q "huggingface_hub>=1.5.0,<2.0"
    !pip install -q torchcodec
    !pip install -q --no-deps --upgrade timm
else:                                    # qwen3-4b
    !pip install -q pip3-autoremove
    !pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
    !pip install -q unsloth
    !pip install -q --no-deps --upgrade "torchao>=0.16.0"
    !pip install -q transformers==4.56.2
    !pip install -q --no-deps trl==0.22.2
```

Drei Punkte, die man verstanden haben sollte:

**Die Pins stammen aus Unsloths eigenen Kaggle Notebooks**, nicht aus Vermutungen. Jede
Modellfamilie hat eine Kombination, die auf diesem Image nachweislich funktioniert;
Abweichungen erzeugen Importfehler oder stille Versionskonflikte.

**`--no-deps` ist Absicht.** Es installiert ein Paket, ohne pip „hilfreich" benachbarte
Pakete neu auflösen und herabstufen zu lassen. Ohne diese Option kann die Installation von
`trl` `transformers` auf eine andere Version ziehen und die Unsloth Patches zerstören.

**Gemma und Qwen3 brauchen unvereinbare Stacks**: `transformers 5.10.1` gegen `4.56.2`.
Sie können nicht koexistieren; deshalb ändert der Preset Schalter auch die
Installationszelle. Schon das ist ein guter Grund, pro Notebook eine Modellfamilie zu
wählen.

Die Installation dauert rund 5 Minuten und muss einmal pro Sitzung laufen.

### Die Umgebung prüfen

Unsloth gibt beim Import ein Banner aus. Lesen Sie es: es bestätigt, was wirklich geschah:

```
==((====))==  Unsloth 2026.9.4: Fast Qwen3 patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.562 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
```

`Num GPUs = 2` bestätigt, dass die Beschleunigereinstellung griff. `Bfloat16 = FALSE`
bestätigt die T4 Randbedingung. `Fast Qwen3 patching` bestätigt, dass Unsloth die
Architektur erkannt hat. Stimmt eine dieser Zeilen nicht, hören Sie hier auf, statt später
das Training zu debuggen.

---

## Beide GPUs nutzen

Es gibt zwei Wege, eine zweite T4 einzusetzen, und der hier voreingestellte ist der
weniger naheliegende.

### Standard: zwei verschiedene Jobs parallel

Ein 4B Modell mit QLoRA passt auf **eine** T4. Einen Job auf beide Karten aufzuteilen
bringt bei dieser Größe wenig und erhöht das Koordinationsrisiko. Stattdessen trainiert
GPU 0, während GPU 1 Basisantworten des untrainierten Modells erzeugt, Antworten, die man
ohnehin braucht, um zu belegen, dass das Fine Tuning etwas bewirkt hat.

Der Mechanismus besteht aus einfachen Subprozessen mit fest zugewiesener Karte:

```python
def run(cmd, gpus, name, background=False):
    """Skript auf den angegebenen GPUs im eigenen Ordner ausführen."""
    cwd = f"{WORK}/run_{name}"
    os.makedirs(cwd, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "PYTHONPATH": SCRIPTS}
    if background:
        log = open(f"{cwd}/log.txt", "w")
        return subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    ...
```

```python
baseline = run(baseline_cmd, "1", "baseline", background=True)   # GPU 1
run([PY, *train_args], "0", "train")                             # GPU 0
code = baseline.wait()
```

Zwei leicht zu übersehende Details:

**Jeder Prozess erhält ein eigenes Arbeitsverzeichnis.** Unsloth schreibt einen
Kernel Cache in das aktuelle Verzeichnis. Zwei Prozesse mit gemeinsamem Arbeitsverzeichnis
geraten in ein Wettrennen um diesen Cache. Das ist nicht theoretisch: genau deshalb
existiert `cwd=f"{WORK}/run_{name}"`.

**`CUDA_VISIBLE_DEVICES` wird je Subprozess gesetzt, nicht global.** Jeder Prozess sieht
genau eine Karte und nennt sie `cuda:0`. Kein Code im Trainingsskript kennt GPUs überhaupt.

### Optional: echtes verteiltes Training

```python
run([PY, "-m", "torch.distributed.run", "--nproc_per_node=2", *train_args], "0,1", "train")
```

Das funktioniert und wurde für das vollständige SmolLM2 Fine Tuning genutzt. Es erzeugte
zugleich den schlimmsten Fehlschlag des Projekts.

---

## Der Fehlschlag, für den man planen sollte

Der SmolLM2 Lauf absolvierte **alle 51 Trainingsschritte auf beiden T4 erfolgreich** und
stürzte dann ab:

```
NCCL ALLGATHER sequence number mismatch
```

Er starb *nach* dem Training, beim Laden des besten Checkpoints für den Export. Das
Training war in Ordnung; die kollektive Koordination im Exportpfad nicht.

Gerettet hat den Lauf, dass **Kaggle `/kaggle/working` auch bei einem gescheiterten
Notebook als Ausgabe erhält.** Alle 41 Ausgabedateien einschließlich der Checkpoints
überlebten. Die Lösung war ein separates Einzelprozess Notebook, das den erhaltenen
Checkpoint lud, auswertete und exportierte, ganz ohne Kollektive. Es lief in 218 Sekunden.

**Drei Regeln folgen daraus:**

1. **Export gehört nicht in denselben Prozess wie verteiltes Training.** Trainieren,
   Checkpoints speichern, beenden. Export getrennt.
2. **Jede Epoche einen Checkpoint schreiben.** Erst das macht eine Rettung möglich.
3. **Ein gescheitertes Kaggle Notebook ist kein verlorener Lauf.** Prüfen Sie die Ausgabe,
   bevor Sie etwas neu starten: der teure Teil liegt vielleicht schon auf der Platte.

---

## Automatisierung vom Terminal

Für einen einzelnen Lauf genügt die Weboberfläche. Für eine Versuchsreihe ist die
Kaggle API deutlich besser, und erlaubt alles ohne Browser.

```bash
pip install kaggle
# Token unter kaggle.com/settings -> API -> Create New Token
# wird in ~/.kaggle/access_token abgelegt
```

### Einen Datensatz hochladen

```bash
# dataset-metadata.json im Ordner neben den Datendateien:
# { "title": "Shahnameh Style SFT",
#   "id": "<benutzername>/shahnameh-style-sft",
#   "licenses": [{"name": "CC0-1.0"}] }

kaggle datasets create -p .        # standardmäßig privat
```

### Ein Notebook hochladen und ausführen

`kernel-metadata.json` steuert alles, was sonst das rechte Bedienfeld tut:

```json
{
  "id": "<benutzername>/shahnameh-qwen3-4b-unsloth",
  "title": "Shahnameh Qwen3-4B Unsloth",
  "code_file": "shahnameh-qwen3-4b-unsloth.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<benutzername>/shahnameh-style-sft"],
  "kernel_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
```

**`"machine_shape": "NvidiaTeslaT4"` wählt T4 × 2 aus.** `enable_gpu: true` allein kann
eine einzelne Karte liefern. Dieses Feld ist schlecht dokumentiert; der verlässliche Weg
zum richtigen Wert ist, die Metadaten eines bereits korrekt konfigurierten Notebooks zu
holen:

```bash
kaggle kernels pull <benutzername>/<vorhandenes-notebook> -m -p .
```

Danach:

```bash
kaggle kernels push -p .                          # Hochladen startet zugleich den Lauf
kaggle kernels status <benutzername>/<notebook>   # PENDING / RUNNING / COMPLETE / ERROR
kaggle kernels output <benutzername>/<notebook> -p ./out
```

Eine einfache Abfrageschleife genügt meist:

```bash
while true; do
  S=$(kaggle kernels status <benutzername>/<notebook> | grep -o 'KernelWorkerStatus\.[A-Z_]*')
  echo "$(date +%H:%M:%S) $S"
  case "$S" in *COMPLETE*|*ERROR*) break;; esac
  sleep 120
done
```

### Notebooks verketten

Die Ausgabe eines Notebooks kann die Eingabe eines anderen sein, über `kernel_sources`.
Dieses Projekt nutzte das, um einen Decoding Vergleich gegen einen bereits trainierten
Adapter zu fahren, ohne neu zu trainieren:

```json
"kernel_sources": ["<benutzername>/shahnameh-qwen3-4b-fa-only"]
```

---

## Stolperfallen, die echte Zeit gekostet haben

Alle folgenden Punkte traten in diesem Projekt auf. Keiner steht in der offiziellen
Dokumentation.

**Notebook Ausgaben werden unter einem längeren Pfad eingehängt als erwartet.** Ein über
`kernel_sources` angehängtes Notebook erscheint *nicht* unter `/kaggle/input/<slug>/`. Der
tatsächliche Pfad lautet:

```
/kaggle/input/notebooks/<benutzername>/<slug>/...
```

Den kurzen Pfad fest zu verdrahten erzeugt `Can't find 'adapter_config.json'` und einen
Abbruch nach rund 160 Sekunden. Suchen statt annehmen:

```python
found = [os.path.dirname(p)
         for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
         if "checkpoint" not in p]
```

**Die Kaggle CLI verstümmelt absolute Windows Pfade.**
`kaggle datasets create -p "C:/langer/pfad"` baut einen fehlerhaften temporären Pfad und
scheitert mit `[Errno 2] No such file or directory`. Abhilfe: aus dem Ordner heraus mit
`-p .` aufrufen.

**Notebooks mit Nicht ASCII Zeichen lassen sich unter Windows nicht hochladen.** Ein
Notebook mit persischem Text löst `'charmap' codec can't decode byte 0x81` aus. UTF-8
erzwingen:

```bash
PYTHONUTF8=1 kaggle kernels push -p .
```

**`enable_gpu: true` bedeutet nicht zwei GPUs.** Ohne `machine_shape` bekommt ein
Zwei GPU Notebook womöglich stillschweigend eine Karte, und jeder Code mit
`CUDA_VISIBLE_DEVICES=1` schlägt fehl.

**Eine gespeicherte Notebook Seite ist schreibgeschützt.** Um mit einem Notebook zu
interagieren (eine Frage in ein Eingabefeld tippen, eine Zelle ausführen), muss man auf
**Edit** klicken und eine aktive Sitzung starten. Die gespeicherte Version anzusehen und
Interaktivität zu erwarten, ist eine verbreitete Verwechslung.

**Interaktive Sitzungen laufen ab.** Installationszellen und Ladezellen müssen nach einem
Timeout erneut ausgeführt werden. Rechnen Sie mit rund 7 Minuten bis zur ersten
brauchbaren Ausgabe.

---

## Kostenübersicht dieses Projekts

| Lauf | Laufzeit | Ergebnis |
|---|---|---|
| SmolLM2 vollständiges Fine Tuning (2× T4, DDP) | ~25 Min. | trainiert, dann Absturz beim Export |
| SmolLM2 Rettung/Export | 218 s | das nutzbare Modell |
| Qwen3-4B mehrsprachiges LoRA | 411 s Training | 33 % persische Reimquote |
| Qwen3-4B nur Persisch, LoRA | 724 s Training | 65 % persische Reimquote |
| Persischer Decoding Vergleich | ~12 Min. | 5 Konfigurationen gemessen |
| **Summe** | **unter 3 Stunden** | von 30 Wochenstunden |

Ein Modell mit 4 Milliarden Parametern feinabzustimmen liegt klar innerhalb dessen, was
kostenlose Rechenzeit leisten kann. Der begrenzende Faktor waren in diesem Projekt nie die
GPUs, sondern die lediglich 90 persischen Trainingsbeispiele. Was das beheben würde, steht
in der [Haupt README](../README.de.md).
