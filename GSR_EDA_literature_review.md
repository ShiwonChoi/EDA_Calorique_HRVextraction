# EDA/GSR preprocessing — literature review

Prepared to ground the GSR/EDA extraction pipeline described in
`GSR_EDA_extraction_plan.md`. Citations were checked against CrossRef, OpenAlex,
Semantic Scholar, and (where possible) primary/publisher sources by a dedicated
research pass — verification status is marked per source. Where evidence was
mixed, ambiguous, or simply not findable, that is flagged explicitly rather than
smoothed over.

## 1. Sampling rate for EDA analysis

The Shimmer3 GSR+ records natively at **64 Hz**; this pipeline's shared
resampling step (used identically for the co-recorded PPG channel) upsamples to
250 Hz via Akima interpolation before any EDA-specific processing.

64 Hz clears the generic threshold for tonic/SCL-level work (commonly cited
minimum ≈10 Hz) and is a common rate among wearable-tier EDA devices, several of
which run lower (e.g. 32 Hz) and are flagged in the literature as suboptimal for
precise SCR work. It sits below the ≥200 Hz (with 1–2 kHz common in lab-grade
desktop systems) some guidance recommends specifically for precise SCR
onset/rise-time timing. Net assessment: **adequate for SCL/tonic analysis and
non-specific SCR counting; not ideal for high-precision SCR latency work.**

Upsampling via interpolation before decomposition is a neutral step for
alignment purposes — it does not synthesize new high-frequency information, so
it does not "fix" the native rate's ceiling on timing precision. The literature
does not directly endorse Akima interpolation by name for EDA specifically (it
is well-established for PPG); this is an inference from general
resampling/interpolation properties, not a directly-cited EDA claim — flagged
as a gap, not asserted as literature consensus.

**Sources:**
- Society for Psychophysiological Research Ad Hoc Committee on Electrodermal
  Measures (Boucsein, W., Fowles, D.C., Grimnes, S., Ben-Shakhar, G., Roth,
  W.T., Dawson, M.E., & Filion, D.L.) (2012). Publication recommendations for
  electrodermal measurements. *Psychophysiology*, 49(8), 1017–1034.
  https://doi.org/10.1111/j.1469-8986.2012.01384.x — verified via CrossRef.
  [peer-reviewed]
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
  https://doi.org/10.1007/978-1-4614-1126-0 — verified via publisher/ISBN
  lookup. [book, peer-reviewed academic monograph]
- Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. (2015).
  Automatic identification of artifacts in electrodermal activity data.
  *EMBC 2015*. https://doi.org/10.1109/embc.2015.7318762 — verified via
  CrossRef. [conference paper] (contextual: wearable-rate limitations)

## 2. Known Shimmer GSR-specific artifacts

Two concrete artifacts were observed directly in this project's raw data
(`Data/SC_03/.../shimmer_P003_Trial00_baseline_20260205_143554.csv`): implausible
negative resistance values (e.g. -983 kOhm) and runs of exactly-repeated values.

**Could not verify:** a specific peer-reviewed paper on Shimmer3 GSR+
range-switching artifacts (an initial "Bari/Sanches" lead was checked
extensively and could not be substantiated in CrossRef/OpenAlex/Semantic
Scholar). This is flagged here explicitly rather than fabricated.

**What is documented:** Shimmer's own technical documentation describes 4
auto-selectable resistance ranges (encoded in the top 2 bits of the raw ADC
word); during a range change the firmware holds/duplicates the last valid value
through an ~80 ms settling window. This directly explains the repeated-value
plateaus as a documented firmware behavior, not random noise. The negative
resistance values are consistent with mis-decoded/out-of-range ADC codes during
range transitions or electrode disconnection — this is a reasoned inference
from the auto-range documentation plus general EDA artifact literature, not a
verbatim claim from any single paper.

**Sources:**
- Shimmer GSR User Guide — official Shimmer technical documentation (primary
  source, not peer-reviewed, but authoritative for hardware behavior).
- Kleckner, I.R., Jones, R.M., Wilder-Smith, O., Wormwood, J.B., Akcakaya, M.,
  Quigley, K.S., Lord, C., & Goodwin, M.S. (2018). Simple, transparent, and
  flexible automated quality assessment procedures for ambulatory electrodermal
  activity data. *IEEE Transactions on Biomedical Engineering*, 65(7),
  1460–1467. https://doi.org/10.1109/tbme.2017.2758643 — verified via CrossRef.
  [peer-reviewed] — source of the 0.05–60 µS plausible-range and ~10 µS/s
  max-slope artifact heuristics used in the plan.
- Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. (2015).
  Automatic identification of artifacts in electrodermal activity data.
  *EMBC 2015*. https://doi.org/10.1109/embc.2015.7318762 — verified via
  CrossRef. [conference paper] — EDA Explorer tool, ML-based artifact
  classification.
- Chen, W., Jaques, N., Taylor, S., Sano, A., Fedor, S., & Picard, R.W. (2015).
  Wavelet-based motion artifact removal for electrodermal activity. *EMBC
  2015*. https://doi.org/10.1109/embc.2015.7319814 — verified via CrossRef.
  [conference paper]
- Kelsey, M., Akcakaya, M., Kleckner, I.R., Palumbo, R.V., Barrett, L.F.,
  Quigley, K.S., & Goodwin, M.S. (2018). Applications of sparse recovery and
  dictionary learning to enhance analysis of ambulatory electrodermal activity
  data. *Biomedical Signal Processing and Control*, 40, 58–70.
  https://doi.org/10.1016/j.bspc.2017.08.024 — verified via CrossRef.
  [peer-reviewed]
- Burns, A., Doheny, E.P., Greene, B.R., Foran, T., Leahy, D., O'Donovan, K., &
  McGrath, M.J. (2010). SHIMMER™: An extensible platform for physiological
  signal capture. *EMBC 2010*. https://doi.org/10.1109/iembs.2010.5627535 —
  verified via CrossRef/OpenAlex. [conference paper] — the standard Shimmer
  platform validation citation; does not itself address range-switching
  artifacts.

## 3. Unit conversion (resistance → conductance)

`Conductance (µS) = 1000 / Resistance (kOhm)`. Conductance is the
near-universally preferred analysis domain: SCR amplitude/morphology is
approximately linear in conductance but nonlinear (compressed at low
resistance, expanded at high resistance) in resistance — sweat gland activity
maps onto conductance more directly.

**Sources:**
- Dawson, M.E., Schell, A.M., & Filion, D.L. (2016). The electrodermal system.
  In J.T. Cacioppo, L.G. Tassinary, & G.G. Berntson (Eds.), *Handbook of
  Psychophysiology* (4th ed., pp. 217–243). Cambridge University Press.
  https://doi.org/10.1017/9781107415782.010 — verified via Cambridge Core
  listing. [book chapter, peer-reviewed academic volume]
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer. (as
  above) — explicit discussion of conductance-vs-resistance domain choice.
- SPR committee report (Boucsein et al. 2012), as above — reiterates
  conductance as the preferred reporting unit.

## 4. Tonic/phasic decomposition methods

Ranking, current best-practice vs. legacy:

- **Most current/most-cited model-based methods:** cvxEDA (Greco et al. 2016)
  and CDA/Ledalab (Benedek & Kaernbach 2010, *J Neurosci Methods*) are the two
  most widely used model-based decomposition approaches in modern
  psychophysiology and affective computing.
- **DDA** (Benedek & Kaernbach 2010, *Psychophysiology* — nonnegative
  deconvolution) is the discrete-event predecessor to CDA within the same
  Ledalab toolbox family; largely superseded by CDA for continuous analysis but
  still used for discrete-trial designs.
- **Sparse deconvolution** (Hernando-Gallego et al. 2018, *IEEE JBHI*) is a
  newer, less-adopted, growing approach.
- **Simple high-pass/moving-average methods** (Biopac/AcqKnowledge
  convention) are legacy/proprietary-software-style approaches — fast and
  simple, still common in applied/wearable-computing contexts, but not
  considered best practice for rigorous psychophysiology work.

**NeuroKit2 method ↔ paper mapping** (verified directly against NeuroKit2
source docstrings):
- `highpass` / `neurokit` / `biopac` / `acqknowledge` → Butterworth high-pass
  (phasic) / low-pass (tonic), 0.05 Hz default cutoff; attributed to Biopac
  AcqKnowledge convention — **no independent peer-reviewed origin** (vendor
  documentation, not academic literature).
- `smoothmedian` / `median` → median-smoothing subtraction; also attributed to
  Biopac AcqKnowledge convention (same caveat).
- `cvxeda` / `convex` → Greco, A., Valenza, G., Lanata, A., Scilingo, E.P., &
  Citi, L. (2016). cvxEDA: A convex optimization approach to electrodermal
  activity processing. *IEEE Transactions on Biomedical Engineering*, 63(4),
  797–804. https://doi.org/10.1109/tbme.2015.2474131 — verified via
  CrossRef/OpenAlex. (Note: DOI registered as 2015 early-access; print volume
  63(4) is 2016 — cite as 2016 per print volume if consistency matters.)
  [peer-reviewed]
- `sparse` / `sparseda` → Hernando-Gallego, F., Luengo, D., & Artés-Rodríguez,
  A. (2018). Feature extraction of galvanic skin responses by nonnegative
  sparse deconvolution. *IEEE Journal of Biomedical and Health Informatics*,
  22(5), 1385–1394. https://doi.org/10.1109/jbhi.2017.2780252 — verified via
  CrossRef/OpenAlex (same early-access-vs-print-year note: online 2017, print
  2018). [peer-reviewed] — **flagged by NeuroKit2's own documentation as
  experimental/needing further validation.**

**Core decomposition citations:**
- Benedek, M., & Kaernbach, C. (2010). A continuous measure of phasic
  electrodermal activity. *Journal of Neuroscience Methods*, 190(1), 80–91.
  https://doi.org/10.1016/j.jneumeth.2010.04.028 — verified via CrossRef.
  [peer-reviewed] (CDA)
- Benedek, M., & Kaernbach, C. (2010). Decomposition of skin conductance data
  by means of nonnegative deconvolution. *Psychophysiology*, 47(4), 647–658.
  https://doi.org/10.1111/j.1469-8986.2009.00972.x — verified directly via
  CrossRef API (251 citations at time of check). [peer-reviewed] (DDA)
- Makowski, D., Pham, T., Lau, Z.J., Brammer, J.C., Lespinasse, F., Pham, H.,
  Schölzel, C., & Chen, S.H.A. (2021). NeuroKit2: A Python toolbox for
  neurophysiological signal processing. *Behavior Research Methods*, 53(4),
  1689–1696. https://doi.org/10.3758/s13428-020-01516-y — verified via
  OpenAlex. [peer-reviewed]

**Not found / omitted:** "PhasoR" was searched for as a possible named method
and could not be confirmed as an established citable method in the EDA
decomposition literature — omitted from the plan rather than included on
unverified grounds.

CDA/Ledalab itself is not available as a maintained Python package (it is a
MATLAB toolbox) — implementing it would require a separate port, out of scope
for this pass and noted as future work in the plan.

## 5. SCR/peak detection criteria and standard metrics

Two amplitude-threshold conventions coexist depending on subfield: (a) a
legacy/paper-chart-era convention of **0.05 µS**, still common in clinical/some
psychophysiology-lab work, and (b) a modern convention allowing smaller
**0.01 µS** thresholds for higher-resolution digital systems.

**Rise-time units — unresolved, flagged rather than silently picked.** Every
accessible secondary source quoting the Braithwaite guide repeats "rise time
range of 0.1–5 msec" verbatim; direct text extraction from the primary PDF
failed (binary stream, not parseable in this pass). 0.1–5 *milliseconds* is not
physiologically plausible for an SCR — established literature (Boucsein 2012;
Dawson, Schell & Filion 2016) puts SCR rise time at roughly 1–3 *seconds*. This
is very likely a transcription/OCR error (sec→msec) propagated across
secondary citations of the guide, but this could not be fully confirmed against
the primary source's original wording. Both figures are presented here with
this caveat rather than one being silently chosen.

Standard per-window/per-trial metrics reported in the literature: SCR
frequency/rate (responses per minute), SCR amplitude (mean and/or sum of
qualifying responses), SCL mean level, SCL slope/habituation trend, area under
the phasic curve, and non-specific SCR (NS-SCR) rate as a tonic-arousal proxy.
These map directly onto the metrics specified in `GSR_EDA_extraction_plan.md`.

**Sources:**
- SPR committee report (Boucsein et al. 2012), *Psychophysiology*, 49(8),
  1017–1034 (same DOI as Section 1) — authoritative SPR-endorsed guidance on
  thresholds and reporting metrics. [peer-reviewed]
- Braithwaite, J.J., Watson, D.G., Jones, R., & Rowe, M. (2013). *A Guide for
  Analysing Electrodermal Activity (EDA) & Skin Conductance Responses (SCRs)
  for Psychological Experiments*. Technical report/guide, Selective Attention &
  Awareness Laboratory (SAAL), Behavioural Brain Sciences Centre, University of
  Birmingham, UK.
  https://www.birmingham.ac.uk/documents/college-les/psych/saal/guide-electrodermal-activity.pdf
  — **citation-hygiene flag:** multiple secondary sources mis-cite this as
  "*Psychophysiology*, 49, 1017–1034" — that citation actually belongs to the
  Boucsein et al. (2012) SPR committee report (Section 1/above). Braithwaite et
  al. (2013) is a lab technical report/guide, **not** a peer-reviewed journal
  article, and was not found indexed in CrossRef, OpenAlex, or Semantic
  Scholar as a journal publication. Cited here explicitly as
  `[technical report/guide — not peer-reviewed]`; flagging the miscitation
  pattern so it doesn't get propagated further.

## Limitations of this review

- No Shimmer3-GSR+-specific peer-reviewed artifact paper was found (Section 2)
  — the artifact-handling recommendation is a synthesis of hardware
  documentation and general (non-Shimmer-specific) EDA quality literature.
- The Braithwaite et al. (2013) guide's SCR rise-time figure could not be
  confirmed against primary-source text (Section 5) — treat the "0.1–5 msec"
  figure commonly quoted online with suspicion; "seconds" is more consistent
  with the rest of the literature.
- "PhasoR" and the "Bari/Sanches" Shimmer-artifact paper were both searched
  for and could not be verified — omitted rather than fabricated.
- The two IEEE decomposition papers (cvxEDA, sparse deconvolution) each have a
  DOI early-access year one year earlier than their print-volume year; this
  review cites the print-volume year for both, consistent with how they are
  most commonly referenced.