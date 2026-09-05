# Erklärbare KI: Vorhersage des Beschäftigungswachstums 2024–2034 anhand von O*NET-Berufsmerkmalen

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> 🇬🇧 English version: [`README.md`](README.md)

Ein einzelnes, in sich geschlossenes Jupyter-Notebook mit einer einfachen Leitfrage:

**Lässt sich die vom Bureau of Labor Statistics (BLS) prognostizierte Beschäftigungsänderung eines Berufs (2024–2034) daraus vorhersagen, was dieser Beruf tatsächlich ausmacht — also aus den Fähigkeiten, dem Wissen, den Aktivitäten, dem Arbeitskontext und den Arbeitsstilen, die O\*NET für ihn erfasst?**

Das Notebook führt zwei unabhängige öffentliche Datensätze zusammen, trainiert einen Random-Forest-Regressor, vergleicht ihn mit einer Mittelwert-Baseline und einem regularisierten linearen Modell und macht anschließend mit **SHAP (SHapley Additive exPlanations)** sowohl das globale Verhalten des Modells als auch einzelne Vorhersagen nachvollziehbar.

---

## Inhaltsverzeichnis

- [Forschungsfrage](#forschungsfrage)
- [Ergebnisse auf einen Blick](#ergebnisse-auf-einen-blick)
- [Datenquellen](#datenquellen)
- [Projektstruktur](#projektstruktur)
- [Einstieg](#einstieg)
- [Was das Notebook Schritt für Schritt tut](#was-das-notebook-schritt-für-schritt-tut)
- [Modellierung im Detail](#modellierung-im-detail)
- [Erklärbarkeit mit SHAP](#erklärbarkeit-mit-shap)
- [Wichtige Designentscheidungen](#wichtige-designentscheidungen)
- [Grenzen der Analyse](#grenzen-der-analyse)
- [Quellen](#quellen)

---

## Forschungsfrage

> Welche O\*NET-Berufsmerkmale hängen am stärksten mit der vom BLS prognostizierten Beschäftigungswachstumsrate 2024–2034 zusammen, und wie lässt sich der Beitrag einzelner Merkmale zu einer konkreten Modellvorhersage mit SHAP erklären?

Die Fragestellung ist bewusst **auf Zusammenhänge und nicht auf Kausalität** ausgerichtet. Das Modell lernt Muster, die Berufsprofile mit BLS-Prognosen verknüpfen; es erstellt keine eigenständige Vorhersage über den Arbeitsmarkt.

---

## Ergebnisse auf einen Blick

Bewertet auf einem zurückgehaltenen Testset von 20 % (155 Berufe); das Kreuzvalidierungs-R² ist ein 5-facher Score ausschließlich auf den Trainingsdaten.

| Modell | MAE (PP) | RMSE (PP) | R² (Test) | CV-R² (Train) |
|---|---:|---:|---:|---:|
| Baseline (sagt den Mittelwert vorher) | 5,118 | 7,135 | −0,009 | −0,001 |
| Ridge Regression (`RidgeCV`, α = 221,2) | 4,353 | 5,928 | 0,303 | 0,349 |
| **Random Forest (getunt)** | **4,149** | **5,834** | **0,325** | **0,352** |

*MAE/RMSE in Prozentpunkten der prognostizierten Beschäftigungsänderung.*

**Interpretation:** Rund ein Drittel der Varianz in den BLS-Wachstumsprognosen lässt sich allein aus Berufsmerkmalen rekonstruieren. Das ist ein echtes, nicht triviales Signal — die Baseline erklärt gar nichts — aber weit entfernt von einem gelösten Vorhersageproblem, und das Baumensemble schlägt ein gut regularisiertes lineares Modell nur knapp. Beide Modelle ziehen ihre Vorhersagen systematisch zum Mittelwert, sodass stark wachsende Ausreißer deutlich unterschätzt werden:

| Beruf | Tatsächlich | Vorhergesagt | Abs. Fehler |
|---|---:|---:|---:|
| Data scientists | 33,5 % | 5,98 % | 27,52 |
| Operations research analysts | 21,5 % | 4,75 % | 16,75 |
| Adult basic education, adult secondary e… (instructors) | −13,7 % | 2,07 % | 15,77 |
| Computer numerically controlled tool programmers | 12,8 % | −2,29 % | 15,09 |
| Nuclear power reactor operators | −15,3 % | −0,42 % | 14,88 |

Die Zielvariable selbst reicht von **−36,1 % bis +49,9 %**.

---

## Datenquellen

Zwei unabhängige öffentliche Datensätze, verknüpft über den 6-stelligen SOC-Code.

### 1. O\*NET — Berufsmerkmale (`datasets/*.xlsx`)

Die O\*NET-Datenbank beschreibt jeden Beruf über hunderte bewertete Merkmale. Die Dateien liegen im *long format* vor (eine Zeile pro Beruf × Element × Skala) und werden im Notebook ins *wide format* überführt.

| Datei | verwendete Scale ID | Bedeutung | erzeugte Merkmale | Prefix |
|---|---|---|---:|---|
| `Abilities.xlsx` | `IM` | Importance | 52 | `ability_` |
| `Essential Skills.xlsx` | `IM` | Importance | 10 | `skill_` |
| `Knowledge.xlsx` | `IM` | Importance | 33 | `knowledge_` |
| `Work Activities.xlsx` | `IM` | Importance | 41 | `activity_` |
| `Work Context.xlsx` | `CX` | Kontext-Rating (einzige numerische Skala der Datei) | 55 | `context_` |
| `Work Styles.xlsx` | `WI` | Work Styles Impact | 21 | `style_` |
| `Job Zones.xlsx` | — | bereits wide; Vorbereitungsniveau 1–5 | 1 | `job_zone` |
| `Occupation Data.xlsx` | — | Berufsbezeichnungen und -beschreibungen (Join-Metadaten) | — | — |

**→ 213 O\*NET-Merkmale über 923 O\*NET-SOC-Codes.**

### 2. BLS Employment Projections (`datasets/occupation.xlsx`, Blatt `Table 1.2`)

Die National Employment Matrix 2024–34. Es werden nur Zeilen mit `Occupation type == "Line item"` behalten (Einzelberufe, keine Summenzeilen wie *„Total, all occupations"*).

- **Zielvariable:** `Employment change, percent, 2024–34`
- **Behaltene Basisjahr-Kovariaten:** Beschäftigung 2024, Medianlohn 2024, Anteil Selbstständiger 2024, typische Eingangsqualifikation, geforderte Berufserfahrung, typische Einarbeitung.

---

## Projektstruktur

```
explainable_ai/
├── index.ipynb          # Die gesamte Analyse: Laden → Bereinigen → Modellieren → SHAP
├── requirements.txt     # Gepinnte Abhängigkeiten
├── datasets/
│   ├── Abilities.xlsx           # O*NET: 52 Fähigkeits-Ratings je Beruf
│   ├── Essential Skills.xlsx    # O*NET: 10 Basic Skills
│   ├── Knowledge.xlsx           # O*NET: 33 Wissensdomänen
│   ├── Work Activities.xlsx     # O*NET: 41 generalisierte Arbeitsaktivitäten
│   ├── Work Context.xlsx        # O*NET: 55 physische/soziale Kontext-Ratings
│   ├── Work Styles.xlsx         # O*NET: 21 Arbeitsstil-Ratings
│   ├── Job Zones.xlsx           # O*NET: Job Zone 1–5
│   ├── Occupation Data.xlsx     # O*NET: Titel + Beschreibungen
│   └── occupation.xlsx          # BLS Employment Projections, Table 1.2
├── README.md            # Englische Fassung
└── README.de.md         # Diese Datei (Deutsch)
```

Es gibt bewusst kein `src/`-Paket — die Analyse ist als ein einziges Notebook angelegt, das von oben nach unten gelesen werden soll.

---

## Einstieg

**Voraussetzung:** Python 3.11+ (entwickelt mit 3.14).

```bash
git clone https://github.com/<dein-user>/explainable_ai.git
cd explainable_ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab index.ipynb
```

Anschließend alle Zellen von oben nach unten ausführen. Alle Daten liegen im Repository, es sind keine Downloads nötig. Jede zufällige Operation ist mit `RANDOM_STATE = 42` gesetzt, die oben genannten Zahlen sind also exakt reproduzierbar.

**Hinweis zur Laufzeit:** Die `RandomizedSearchCV` in Abschnitt 7 trainiert 20 Kandidatenkonfigurationen × 5 Folds mit Wäldern aus 400–800 Bäumen. Auf einem aktuellen Laptop sind einige Minuten einzuplanen. Die Suche läuft mit `n_jobs=1`, der einzelne Wald intern mit `n_jobs=-1`.

---

## Was das Notebook Schritt für Schritt tut

| Abschnitt | Inhalt |
|---|---|
| **1. Setup** | Imports, `RANDOM_STATE = 42`, Plot-Theme. |
| **2.1 O\*NET laden** | `pivot_onet()` liest jede Datei, filtert auf eine einzelne `Scale ID` und pivotiert long → wide (`index="O*NET-SOC Code"`, `columns="Element Name"`, `values="Data Value"`); jede Spalte erhält das Prefix ihrer Quelldomäne. |
| **2.2 Zusammenführen & aggregieren** | Ein Outer Merge verkettet alle sieben Wide-Frames (923 × 215). Der 8-stellige O\*NET-SOC-Code wird auf den 6-stelligen SOC-Code gekürzt, wodurch 923 Codes auf **798** zusammenfallen — Unterberufe (z. B. `11-1011.00` und `11-1011.03`) werden je `soc_code` gemittelt, und zwar **vor** dem BLS-Join. |
| **2.3 BLS laden** | `Table 1.2` wird mit `skiprows=1` eingelesen, auf Line Items gefiltert, Spalten in snake_case umbenannt, numerische Spalten mit `errors="coerce"` bereinigt (die Quelle nutzt „—" für unterdrückte Werte), Zeilen ohne Zielwert entfernt → **832** Berufe. |
| **2.4 Finaler Join** | Inner Merge über `soc_code` → **773 Berufe × 223 Spalten**, per Assertion auf genau eine Zeile pro eindeutigem SOC-Code geprüft. |
| **2.5 / 3. Datenprüfung** | Profil fehlender Werte, Duplikatsprüfung auf vollständigen Zeilen und auf `occupation_title` (beides null), Prüfung des Wertebereichs der Zielvariable. Duplikate werden **geprüft**, statt blind `drop_duplicates()` aufzurufen. |
| **4. Merkmalsdefinition** | Alle 213 O\*NET-Merkmale plus `job_zone` werden behalten, dazu kommen abgeleitete strukturelle Merkmale (siehe unten) → **220 Merkmale** (216 numerisch, 4 kategorial). |
| **5. Split** | `train_test_split(test_size=0.2, random_state=42)` → 618 Train / 155 Test. Erfolgt **bevor** irgendein Imputer oder Scaler gefittet wird. |
| **6. Baseline + Ridge** | `DummyRegressor(strategy="mean")` sowie eine Ridge-Pipeline mit Median-Imputation + Standardisierung (numerisch) und One-Hot-Encoding (kategorial); α wird per `RidgeCV` über `np.logspace(-2, 4, 30)` bestimmt. |
| **7. Random Forest** | Median-Imputation + Ordinal-Encoding, getunt per `RandomizedSearchCV` (20 Ziehungen, 5-fach `KFold`, `scoring="r2"`). |
| **8. Bewertung** | MAE / RMSE / R² auf Test plus CV-R² auf Train für alle drei Modelle; Streudiagramm tatsächlich vs. vorhergesagt; die fünf größten absoluten Fehler mit Berufsbezeichnung. |
| **9. SHAP** | `TreeExplainer` auf dem trainierten Wald → globaler Bar-Plot, Beeswarm-Plot und zwei Einzelfall-Waterfall-Plots. |
| **10. Fazit** | Einordnung und ausdrückliche Vorbehalte zu Kausalität und Prognosegültigkeit. |

---

## Modellierung im Detail

### Merkmalssatz (220 Merkmale)

| Gruppe | Anzahl |
|---|---:|
| `context_` — Arbeitskontext | 55 |
| `ability_` — Fähigkeiten | 52 |
| `activity_` — Arbeitsaktivitäten | 41 |
| `knowledge_` — Wissensdomänen | 33 |
| `style_` — Arbeitsstile | 21 |
| `skill_` — Basic Skills | 10 |
| Struktur (BLS/SOC) | 8 |

Der strukturelle Block liefert Kontext, den reine Merkmals-Ratings nicht abbilden können:

- `log_employment_2024`, `log_median_wage_2024` — log1p-transformiert, da beide Verteilungen stark rechtsschief sind
- `pct_self_employed` — Anteil Selbstständiger 2024
- `education_entry`, `work_experience`, `on_the_job_training` — typische Qualifikationsanforderungen (kategorial)
- `major_group` — die ersten zwei Stellen des SOC-Codes, also die SOC-Berufshauptgruppe, da Prognosen innerhalb einer Gruppe ähnlich ausfallen
- `job_zone` — O\*NET-Vorbereitungsniveau 1–5

### Vorverarbeitung

Beide Modell-Pipelines kapseln die Vorverarbeitung in einem `ColumnTransformer` **innerhalb** der Pipeline, sodass jede Imputations- und Skalierungsstatistik ausschließlich auf Trainings-Folds gefittet wird:

- **Ridge:** Median-Imputation + `StandardScaler` (numerisch), `OneHotEncoder(handle_unknown="ignore")` (kategorial)
- **Random Forest:** Median-Imputation (numerisch), `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` (kategorial) — für Baummodelle ausreichend und sparsamer als One-Hot

Fehlende Werte sind real und ungleich verteilt: `work_experience` fehlt in 660 von 773 Zeilen, `on_the_job_training` in 303, `pct_self_employed` in 190, und jede O\*NET-Spalte in ca. 22 Berufen, die O\*NET nicht erfasst.

### Hyperparameter-Suche

```python
param_distributions = {
    "rf__n_estimators":     [400, 600, 800],
    "rf__max_depth":        [None, 15, 25],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features":     [0.2, 0.33, 0.5, "sqrt"],
}
```

Gewählt: `n_estimators=400`, `max_depth=15`, `min_samples_leaf=4`, `max_features=0.2` — CV-R² **0,352**. Der niedrige `max_features`-Wert ist bei 220 teils redundanten Prädiktoren zu erwarten: Wenige Merkmale je Split dekorrelieren die Bäume.

---

## Erklärbarkeit mit SHAP

Der Random Forest wird mit `shap.TreeExplainer` erklärt, berechnet auf der vorverarbeiteten Testmatrix, damit die Merkmalsnamen in den Plots erhalten bleiben. Drei sich ergänzende Sichten:

1. **Bar-Plot** (`plot_type="bar"`) — mittlerer absoluter SHAP-Wert je Merkmal: welche Merkmale die Vorhersagen am stärksten bewegen, unabhängig von der Richtung.
2. **Beeswarm-Plot** — dieselbe Rangfolge, aber jeder Beruf ist ein Punkt, eingefärbt nach seinem Merkmalswert. Damit wird sichtbar, ob *hohe* (rot) oder *niedrige* (blau) Werte die Vorhersage anheben oder senken.
3. **Waterfall-Plots** — zwei einzelne Berufe, nachverfolgt vom Basiswert des Modells bis zur finalen Vorhersage, einer mit hoher und einer mit niedriger prognostizierter Wachstumsrate. Das Notebook sucht *Data scientists* bzw. *Cashiers* im Testset und weicht auf die höchste/niedrigste Vorhersage aus, wenn sie dort nicht enthalten sind (im eingecheckten Lauf: *Data scientists*, +6,0 %, und *Textile cutting machine setters, operators, and tenders*, −8,7 %).

Die Einordnung im Notebook ist ausdrücklich: Ein weit oben stehendes Merkmal bedeutet, dass **das Modell darauf achtet** und dass Berufe mit ähnlichem Merkmalsprofil in den Daten historisch ähnliche BLS-Prognosen erhalten haben — nicht, dass dieses Merkmal einen Beruf wachsen oder schrumpfen lässt.

---

## Wichtige Designentscheidungen

Diese Entscheidungen wären sonst unsichtbar; jede wird im Notebook selbst begründet.

**Data Leakage wird an drei Stellen vermieden.**
1. `Employment, 2034`, `Employment change, numeric, 2024–34` und `Occupational openings, 2024–34` werden bewusst **nicht** als Merkmale genutzt — sie sind rechnerisch aus der Zielvariable abgeleitet und würden das R² sinnlos aufblähen.
2. Der Train-Test-Split erfolgt, bevor irgendeine Vorverarbeitungsstatistik geschätzt wird.
3. Die Aggregation der Unterberufe erfolgt **vor** dem BLS-Merge, sodass kein Beruf Zeilen auf beiden Seiten eines Duplikats beisteuert.

**Es werden alle verfügbaren Merkmale verwendet, keine Handauswahl.** Eine Vorauswahl weniger „plausibler" Merkmale verwirft Information und ist für die Interpretierbarkeit nicht nötig: SHAP identifiziert die relevanten Merkmale im trainierten Modell ohnehin, und ein Random Forest ist robust gegenüber vielen korrelierten Eingaben.

**Nur die Importance-Skala (IM), nicht Level (LV).** O\*NET bewertet Abilities, Skills, Knowledge und Work Activities auf beiden Skalen. Sie korrelieren sehr stark, und die Aufnahme von LV brachte experimentell keinen Mehrwert.

**Bekannte Abweichungen in den mitgelieferten O\*NET-Dateien.** `Essential Skills.xlsx` enthält nur die 10 *Basic Skills* (Critical Thinking, Active Listening, …), nicht die vollständige O\*NET-Skill-Taxonomie mit Cross-Functional Skills wie *Social Perceptiveness* oder *Complex Problem Solving*. Ebenso nutzt `Work Styles.xlsx` eine etwas andere Elementliste als die Standard-Taxonomie (z. B. kein *Independence*). Das schränkt die Abdeckung der Taxonomie ein, nicht das Vorgehen.

---

## Grenzen der Analyse

- **R² ≈ 0,33 ist eine mäßige Anpassung.** Zwei Drittel der Varianz in den BLS-Prognosen werden durch Berufsmerkmale nicht erklärt. Die Vorhersagen ziehen stark zum Mittelwert, sodass genau die Berufe am schlechtesten getroffen werden, die am interessantesten wären: die extremen Wachser und Schrumpfer.
- **Die Zielvariable ist eine Prognose, kein Ergebnis.** Das Modell wird auf BLS-*Vorhersagen* für 2024–34 trainiert, die auf Annahmen über wirtschaftliche, technologische und demografische Entwicklungen beruhen. Es lernt die Struktur dieser Prognosen, nicht die Zukunft.
- **Korrelation, keine Kausalität.** SHAP-Attributionen beschreiben das Verhalten des Modells, nicht Mechanismen des Arbeitsmarkts.
- **n = 773.** Jede Zeile ist ein Beruf, kein Beschäftigter; für ein Modell mit 220 Merkmalen ist die Stichprobe klein — deshalb tragen Regularisierung und Kreuzvalidierung hier viel Gewicht.
- **Starke Missingness in drei strukturellen Spalten** (`work_experience`, `on_the_job_training`, `pct_self_employed`) wird per Median-Imputation behandelt, was selbst ein Signal enthalten kann, das das Modell ausnutzt.
- **Nur ein Split.** Die Testmetriken stammen aus einem einzigen 80/20-Split mit festem Seed; wiederholte oder verschachtelte Kreuzvalidierung würde die Unsicherheit ehrlicher abbilden.

---

## Quellen

- **O\*NET Resource Center** — Datenbank der Berufsmerkmale: <https://www.onetcenter.org/database.html> (O\*NET-Daten werden vom U.S. Department of Labor unter CC BY 4.0 bereitgestellt)
- **BLS Employment Projections** — National Employment Matrix, 2024–34: <https://www.bls.gov/emp/> (Werk der US-Regierung, gemeinfrei)
- **SHAP** — Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*: <https://github.com/shap/shap>
- **scikit-learn** — <https://scikit-learn.org>

---

## Lizenz

Es liegt noch keine Code-Lizenzdatei bei — vor einer öffentlichen Veröffentlichung sollte eine ergänzt werden. Die mitgelieferten Daten behalten die Bedingungen ihrer ursprünglichen Herausgeber (O\*NET: CC BY 4.0; BLS: gemeinfrei).
