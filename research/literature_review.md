# Literature Review: Scientific Foundations for Artificial Consciousness Research

**Document:** research/literature_review.md
**Date:** 2026-10-13
**Status:** Working review; updated as the project proceeds.
**Scope:** Neuroscience of consciousness, computational models, metacognition and
uncertainty estimation, cognitive architectures, AI agents, and frameworks for
assessing consciousness in artificial systems.

> **Methodological note.** This review separates three epistemic categories:
> (1) **Established evidence** — widely replicated findings or formal results;
> (2) **Contested interpretations** — claims with serious expert disagreement;
> (3) **Speculation** — ideas without direct empirical support. Only verified
> sources are cited. Where a source could not be verified directly, it is marked
> `[unverified]`.

---

## 1. Global Neuronal Workspace Theory (GNWT)

### Core claims
- **Baars (1988)** proposed the Global Workspace Theory: the brain contains many
  parallel specialist processors, most of whose activity is unconscious. A
  limited-capacity workspace selects one winning content at a time and
  broadcasts it globally, making it available to memory, language, and
  decision-making systems. Consciousness, on this account, is the broadcast event.
- **Dehaene & Changeux (formalized 1998 onward; Dehaene & Naccache 2001)**
  gave GWT a neural implementation: the "global neuronal workspace" is realized
  by long-range cortical connections (principally frontoparietal), and
  conscious access corresponds to a nonlinear "ignition" — a sudden, all-or-none
  amplification of the winning representation.
- **Dehaene, Changeux, Naccache, Sackur & Sergent (2006, *PLoS Biology*;
  summarized in Dehaene et al. 2011, *Neuron*, "Experimental and theoretical
  approaches to conscious processing")** developed simulated neural networks
  reproducing masking, the attentional blink, and all-or-none access.
- **Mashour, Ro, Hudetz & Cafarelli (2020, *Neuron*)** review the updated GNWT:
  the workspace is framed as prefrontal working memory supporting conscious
  reasoning over high-level multimodal concepts, linked to Kahneman's "System 2."

### Established evidence
- Subliminal vs. conscious stimulus contrasts consistently show that conscious
  perception recruits widespread frontoparietal activity while unconscious
  processing remains local (Dehaene et al., 2011).
- The P3b ERP component was long treated as a marker of conscious access, but
  **this claim is no longer defended by the theory's proponents** (revised
  2023–2025) — a notable revision often missed in secondary treatments.
- Behavioral signatures (attentional blink, masking thresholds) match the
  theory's predicted all-or-none transition between unconscious and conscious
  perception.

### Contested interpretations
- **The COGITATE adversarial collaboration (Nature, 2025; Vol. 642;
  doi:10.1038/s41586-025-08888-1)** tested IIT and GNWT head-to-head with
  preregistered predictions, n=256 participants, and three imaging modalities
  (fMRI, MEG, iEEG). Result: **neither theory's predictions were fully
  confirmed.** GNWT's predicted ignition at stimulus *offset* was not found, and
  prefrontal decoding of conscious content was weaker than expected; IIT's
  predicted sustained posterior synchronization was not found. Both camps
  dispute the operationalization (Dehaene argues the design did not test
  GNWT's core conscious-vs-unconscious contrast). The experiment does not
  falsify either theory, but it substantially challenges key tenets of both.
- Whether the workspace is strictly prefrontal or a broader frontoparietal
  network remains unsettled; recent formulations emphasize frontoparietal
  networks rather than prefrontal cortex specifically.

### Computational GWT models relevant to this project
- **LIDA (Franklin et al., 2000s+)** — the closest existing serious software
  implementation of GWT. Modules compete via "coalitions"; the winner is
  broadcast periodically; learning is gated by the broadcast. LIDA is
  conceptually strong but the implementation is a dated Java codebase and is
  not LLM-native.
- **Bengio (2017, arXiv:1709.08568, "The Consciousness Prior")** — proposes
  that "conscious" representations in neural networks should be small, sparse,
  and factor-graph-shaped, and that this prior should be built into learning;
  connects the workspace to System-1/System-2 distinctions.
- **VanRullen & Kanai (2021, *Trends in Neurosciences*)** — argue multimodal
  AI systems already exhibit functional analogues of GWT, and that explicit
  GWT-inspired architectures could improve them.
- **Juliani, Kanai & Sasai (2021, *Cognitive Science Society*)** — argue
  DeepMind's Perceiver architecture is a functional global workspace.
- **Goldstein & Kirk-Giannini (2024)** — argue conditionally that if GWT is
  correct, language agents (LLM + memory + planning + tool modules) "might
  easily be made phenomenally conscious if they are not already."
- **Shanahan (2024)** warns against anthropomorphism: what we talk to is a
  character the model plays, not the model itself; LLM self-reports are not
  transparent windows onto machine consciousness.

---

## 2. Integrated Information Theory (IIT)

### Core claims
- **Tononi (2004, 2008); Oizumi, Albantakis & Tononi (2014, IIT 3.0,
  *PLoS Computational Biology*, doi:10.1371/journal.pcbi.1003588);
  Albantakis et al. (2023, IIT 4.0).** Starting from phenomenological axioms,
  IIT derives postulates for the physical substrate of consciousness and
  defines Φ (phi), a measure of integrated information: a system is conscious
  to the extent it generates maximally irreducible cause–effect structure.
- Implications: consciousness is graded, substrate-independent in principle,
  and can in principle be present in simple systems (the theory famously does
  not rule out consciousness in a photodiode or simple logic gates under some
  readings).

### Known limitations (established)
- **Computational intractability:** Φ cannot be computed for systems beyond a
  dozen or so elements (acknowledged in IIT 3.0 itself); it has no operational
  application at brain scale.
- **The exclusion postulate** (only the maximal-Φ complex is conscious) lacks
  independent evidence beyond assertion.
- **COGITATE (2025):** IIT's key dynamic prediction — sustained synchronization
  in posterior cortex during conscious perception — was not found, challenging
  the claim that network connectivity in posterior cortex specifies
  consciousness.

### Contested interpretations
- **September 2023:** 124 scholars signed an open letter characterizing IIT as
  pseudoscience, citing intractability, unfalsifiable structure, and
  counter-intuitive implications. IIT proponents reject these charges; the
  dispute is unresolved.
- Whether Φ identifies *consciousness* or merely *a kind of causal
  integration* is philosophically disputed; critics (e.g., Cerullo 2015) argue
  IIT addresses at best a "pretty hard problem" rather than Chalmers' hard
  problem of consciousness.

### Relevance to this project
- **Butlin et al. (2023) explicitly exclude IIT** from their indicator
  framework because IIT is "incompatible with our working assumption of
  computational functionalism." The project follows Butlin et al. in treating
  IIT as out of scope for the implemented architecture, while retaining its
  conceptual emphasis on integration as a design consideration for the
  workspace (broadcast creates integration, not the other way round).
- IIT's lesson for engineering: a formal measure of causal integration could
  in principle be used as an *internal diagnostic* of workspace coupling, but
  Φ itself is not computable at relevant scale and must not be presented as a
  consciousness meter.

---

## 3. Higher-Order Theories of Consciousness

### Core claims
- A mental state is phenomenally conscious when the subject is *aware of* that
  state — i.e., when it is targeted by a suitable higher-order representation
  (thought or perception). Variants: higher-order thought (Rosenthal),
  higher-order perception (Lycan), higher-order global states.
- **Lau & Rosenthal (2011, *Trends in Cognitive Sciences*)** review empirical
  evidence for higher-order vs. first-order accounts (e.g., Lau & Passingham
  2006 showing dissociations between task performance and subjective
  awareness), finding some evidence favoring higher-order accounts but much
  equivocal.
- **Stanford Encyclopedia of Philosophy** ("Higher-Order Theories of
  Consciousness") provides the canonical taxonomy and objections.

### Engineering translation (Butlin et al. 2023)
Higher-order theories yield four indicator properties for AI systems:
generative/top-down noisy perception, metacognitive monitoring, belief update
driven by metacognition, and sparse smooth coding supporting a quality space.

### Relevance to this project
- Higher-order theories are the closest philosophical ancestor of the
  **metacognition module**: confidence estimation, error detection, and belief
  revision in response to metacognitive signals are direct computational
  analogues. The project treats these as *measurable cognitive functions*,
  not as evidence of phenomenal consciousness.

---

## 4. Recurrent Processing Theory (RPT)

### Core claims
- **Lamme (2006, *Trends in Cognitive Sciences*, 10(5):195–199):** conscious
  visual experience arises from *localized recurrent processing* — feedback
  from higher visual areas to lower ones. The feedforward sweep alone produces
  no experience; *widespread* recurrent processing (toward frontoparietal and
  language areas) produces reportable conscious access.
- Key evidence: backward masking suppresses the delayed recurrent component in
  V1 while leaving the feedforward sweep intact (Lamme et al., 2000); TMS at
  ~100 ms disrupts feedback to V1 and induces blindsight-like effects.
- Lamme distinguishes **phenomenal consciousness** (local recurrence) from
  **access consciousness** (widespread recurrence) — a distinction that
  predicts conscious content *without* reportability.

### Contested / open
- Whether local recurrence is *sufficient* for consciousness, or merely
  necessary, is disputed; recurrent connections also exist in regions not
  associated with consciousness (e.g., V1–thalamus), which the theory must
  explain away.

### Relevance to this project
- RPT motivates **algorithmic recurrence in the architecture**: the workspace
  and metacognition modules must feed back into perception/memory rather than
  forming a pure feedforward pipeline. The experiment battery includes an
  ablation of recurrence (feedforward-only variant) to test whether recurrence
  causally contributes to performance — testing the *functional* claim, not
  the phenomenal one.

---

## 5. Attention Schema Theory (AST)

### Core claims
- **Graziano (2013, *Consciousness and the Social Brain*, Oxford University
  Press):** awareness is the brain's schematic, simplified model of its own
  attentional state. Just as the body schema is a useful-but-inaccurate model
  of the body, the attention schema is a useful-but-inaccurate model of
  attention; its inaccuracy is what makes consciousness feel immaterial.
- AST is **compatible with and complementary to GNWT**: where the workspace
  describes broadcast of selected information, AST describes the system's
  *model of the selection process itself*. Graziano's slogan: "Attention is
  something the brain does; consciousness is something the brain says it has."
- AST is a form of **illusionism**: reports of consciousness are reports of
  the schema's contents, not direct perception of an inner light.

### Relevance to this project
- AST motivates the **attention-monitoring component of the self-model**: the
  system maintains an explicit, inspectable model of what it is attending to
  and why. This is directly implementable and testable (e.g., does an explicit
  attention model improve resource allocation?).
- AST's illusionist reading is a standing caution: fluent self-reports about
  attention or awareness from the system are model outputs, never evidence of
  experience.

---

## 6. Predictive Processing and Active Inference

### Core claims
- **Predictive processing (PP):** the brain is a hierarchical generative model
  that predicts sensory input and updates on prediction error; perception is
  Bayesian inference (Helmholtz → Rao & Ballard 1999; Clark 2013).
- **Friston (2010, *Nature Reviews Neuroscience*, 11:127–138)** formalized the
  **Free Energy Principle**: self-organizing systems minimize variational free
  energy (an upper bound on surprise). **Active inference (Friston et al.,
  2017, "Active Inference: A Process Theory")** unifies perception, action,
  and learning as free-energy minimization: action fulfills predictions,
  perception updates beliefs, and both use precision-weighted prediction
  errors.
- **Precision weighting** (inverse variance of prediction errors) determines
  which errors drive belief updates — the formal analogue of attention.

### Relevance to this project
- Precision weighting provides a principled design for **attention and
  uncertainty integration** in the workspace: reliable signals get high
  precision (drive updates), noisy signals are attenuated.
- Prediction-error-driven learning is the model's first learning mechanism
  (surprise-gated memory updates), without claiming "online learning" in the
  weight-update sense.
- Full variational active inference (e.g., pymdp-style discrete state-space
  planning) is out of scope for the initial build; the project borrows the
  *conceptual* structure (predict → act → observe → update precision-weighted
  beliefs) with standard Bayesian/statistical implementations.

---

## 7. Metacognition, Uncertainty Estimation, and Self-Modeling

### Neuroscience of metacognition (established)
- **Fleming & Lau (2014, *Frontiers in Human Neuroscience*)** distinguish
  three measures: **metacognitive bias** (calibration — systematic
  over/under-confidence), **metacognitive sensitivity** (discrimination of
  correct vs. incorrect decisions), and **metacognitive efficiency**
  (meta-d′/d′, sensitivity corrected for first-order performance).
- **meta-d′** (Maniscalco & Lau, 2012) is the standard SDT-based measure;
  hierarchical Bayesian estimation (Fleming 2017) is the current best practice.
- **Fleming (Annual Review of Psychology review, "Metacognition and
  Confidence")** synthesizes: metacognitive efficiency shows moderate
  test–retest reliability; bias shows stable individual "fingerprints"; both
  relate to mental-health and dogmatism measures.
- Metacognitive ability is substantially **domain-specific** (perceptual vs.
  memory metacognition dissociate neurally); training effects are mixed.

### Uncertainty estimation in LLMs (established)
- **Calibration** (does stated confidence match accuracy?) vs. **sensitivity**
  (does confidence rank correct above incorrect?) are distinct and must be
  measured separately — directly imported from the human literature.
- Methods: token log-probability / predictive entropy (Kadavath et al. 2022);
  **verbalized confidence** (Lin et al. 2022; Tian et al. 2023 — works better
  for RLHF models); **P(True)** self-verification (Kadavath et al. 2022);
  **semantic entropy** (Kuhn, Gal & Farquhar 2023 — clusters by meaning,
  stronger but costlier); self-consistency (Wang et al. 2023).
- **Geng et al. (2024) and Zhou et al. (2024)** survey the field; recent work
  (e.g., Matta et al., 2026 preprint) argues **calibration alone is
  insufficient** — incoherent estimators can appear calibrated — and proposes
  structural coherence and faithfulness metrics.

### Relevance to this project
- The metacognition module implements and separately measures calibration,
  sensitivity, and efficiency on held-out tasks. This is the project's most
  directly measurable research contribution.
- **Self-modeling** in this project means a structured, continuously updated
  representation of *actual system state* (capabilities, limitations,
  resources, recent outcomes, uncertainty) — grounded in tool results and
  logs, never in self-report. Its accuracy is evaluated against ground truth
  (experiment #2 in the battery).

---

## 8. Cognitive Architectures and Modern AI Agents

### Classical architectures (established)
- **Soar** (Laird, Newell & Rosenbloom 1987; Laird 2012, MIT Press):
  production-system architecture with working memory, impasse-driven
  subgoaling, and chunking (compiling deliberation into rules). Still actively
  developed.
- **ACT-R** (Anderson et al. 2004; Ritter, Tehranchi & Oury 2019): modular
  architecture with declarative/procedural memory, subsymbolic activation
  equations (base-level activation, spreading activation, retrieval
  probability). Provides the best-validated quantitative memory math in the
  field.
- **CLARION** (Sun 2007, 2016): dual implicit/explicit representation —
  bottom-level connectionist, top-level symbolic, with bottom-up knowledge
  extraction. Directly prefigures LLM-as-implicit-level designs.
- **LIDA** (Franklin et al.): GWT-based cognitive cycle (see §1).
- **Kotseruba & Tsotsos (2020, *Artificial Intelligence Review*,
  53:17–94)** survey ~86 architectures: ~half lack any real perceptual
  implementation; hybrid symbolic/connectionist designs dominate; there is no
  consensus unified theory.
- **Laird, Lebiere & Rosenbloom (2017, *AI Magazine*)** propose a Common Model
  of Cognition extracting shared commitments (working memory, procedural and
  declarative LTM, perception/action interfaces).

### Modern LLM agents (established)
- **ReAct** (Yao et al. 2022/2023, arXiv:2210.03629): interleaved
  reasoning-and-acting traces.
- **Reflexion** (Shinn et al. 2023, arXiv:2303.11366): verbal self-reflection
  over episodic traces.
- **CoALA** (Sumers, Yao, Narasimhan & Griffiths 2023, arXiv:2309.02427;
  *TMLR* 2024): organizes language agents into memory modules
  (working/episodic/semantic/procedural), an action space (internal vs.
  external actions), and a propose–evaluate–select decision procedure. This is
  the project's closest structural relative.
- **Voyager** (Wang et al. 2023): open-ended skill acquisition with a skill
  library (procedural memory).

### What this project does differently (comparison, not novelty claim)
| Existing work | What it establishes | What this project adds |
|---|---|---|
| CoALA | Vocabulary for describing language agents (memory/action/decision) | An *executable* architecture with a competitive workspace, metacognitive layer, and a falsifiable experiment battery |
| LIDA | GWT-as-software with codelets and broadcast | LLM-native modules, provider abstraction, SQLite persistence, reproducible experiments |
| ReAct/Reflexion | Reasoning traces and verbal self-reflection | *Measured* calibration/sensitivity/efficiency of confidence, not verbal reflection alone |
| ACT-R memory math | Validated activation/decay equations | Borrowed directly for retrieval scoring (with attribution), not reinvented |
| Butlin et al. indicators | Checklist for assessment | A system designed to be *scored* on the checklist plus ablations testing each component's causal contribution |

---

## 9. Indicators and Evaluation of Consciousness in AI

### The Butlin et al. framework (central reference)
- **Butlin, Long, Elmoznino, Bengio, Birch et al. (2023,
  arXiv:2308.08708):** derive 14 **indicator properties** "elucidated in
  computational terms" from RPT, GWT, higher-order theories, PP, and AST,
  under the working assumption of **computational functionalism**. IIT is
  explicitly excluded as incompatible with that assumption.
- The 14 indicators: RPT-1 algorithmic recurrence; RPT-2 integrated perceptual
  representations; GWT-1 parallel specialist systems; GWT-2 limited-capacity
  workspace; GWT-3 global broadcast; GWT-4 state-dependent attention; HOT-1
  generative top-down perception; HOT-2 metacognitive monitoring; HOT-3
  belief revision driven by metacognition; HOT-4 sparse smooth coding (quality
  space); AST-1 predictive model of own attention; PP-1 predictive coding;
  AE-1 agency under competing goals; AE-2 embodiment (output–input
  contingencies).
- Headline conclusion: **"no current AI systems are conscious, but there are
  no obvious technical barriers to building AI systems which satisfy these
  indicators"** — with the explicit caveat that satisfying the indicators
  would *not* mean the system is definitely conscious.
- **Butlin et al. (2025 follow-up, *Trends in Cognitive Sciences*):** reframe
  indicators as inputs to *degrees of belief* about AI consciousness rather
  than yes/no verdicts.

### Complementary frameworks
- **Chalmers (2023, arXiv:2303.07103):** identifies six obstacles to LLM
  consciousness — biology, senses/embodiment, world models, self-models,
  recurrent processing, global workspace, unified agency — and estimates
  <10% credence for current LLMs, ≥25% for future "LLM+" systems within a
  decade. This project directly targets three of the engineering obstacles
  (recurrence, workspace, self-models).
- **Chandaria et al. (2025, arXiv:2609.35618):** a five-level supervenience
  hierarchy with Bayesian aggregation; on identical evidence, optimist and
  skeptic readings of LLM consciousness diverge sharply (0.397 vs. 0.005
  posterior), showing how much verdicts depend on prior commitments.
- **New York Declaration on Animal Consciousness (2024; Birch, Sebo,
  Andrews):** realistic possibility of conscious experience across vertebrates
  and many invertebrates — relevant as a reminder that consciousness science
  is in a credence-based, not verdict-based, state.

### Established cautions
- **Shanahan (2024):** LLM self-reports are character outputs, not
  introspection; anthropomorphism is the default failure mode.
- **Schwitzgebel:** near-future systems may satisfy some theories while
  failing others — single-theory verdicts are unwarranted.
- Indicators are **calibrated to human cognitive markers**, risking false
  positives from imitation (Chandaria et al.).

---

## 10. Known Limitations, Unsolved Problems, and Opportunities

### Known limitations of the field
1. **No agreed operationalization of phenomenal consciousness.** All
   functional indicators risk confusing access-consciousness with
   phenomenal consciousness (Block's distinction).
2. **COGITATE (2025)** partially disconfirmed *both* leading theories'
   signature predictions — the field's best test produced a draw that
   embarrasses confident versions of IIT and GNWT alike.
3. **Φ is uncomputable at scale;** IIT has no operational bridge to brains.
4. **LLM uncertainty estimation** remains brittle: verbalized confidence is
   prompt-sensitive; calibration ≠ coherence; hard factual tasks (e.g.,
   SimpleQA) still show severe miscalibration.
5. **Cognitive architectures** historically under-deliver on perception and
   real-world evaluation (Kotseruba & Tsotsos 2020).

### Unsolved research problems
1. Which functional properties, if any, are *sufficient* for consciousness
   (vs. merely correlated with it)?
2. Can metacognitive monitoring be made *causally effective* (driving belief
   revision and action) rather than merely verbal?
3. How should uncertainty from heterogeneous modules be *integrated* into a
   single decision policy?
4. Does a competitive global workspace causally improve agent performance, or
   is it decorative? (No published ablation answers this for LLM agents.)
5. How to evaluate "self-model accuracy" against ground truth in open-ended
   agents.

### Opportunities for original work (this project's bets)
1. **Ablation-tested workspace + metacognition coupling** in LLM agents:
   no existing system publishes matched-condition ablations of workspace
   integration and metacognitive evaluation against a standard-agent
   baseline on calibration/error-detection metrics. This is the project's
   core experiment (see research_thesis.md).
2. **Self-model accuracy as a measured quantity** with ground-truth
   comparison (capability inventories, resource logs) — a concrete,
   implementable metric where the literature is mostly conceptual.
3. **An open, reproducible experiment battery** (seeds, configs, versioned
   results) mapped explicitly onto Butlin et al.'s 14 indicators plus
   ablations — the field lacks a shared, runnable evaluation harness for
   consciousness-relevant *functional* properties.

---

## 11. Key Sources (verified)

**Theories of consciousness**
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge UP.
- Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of
  consciousness. *Cognition*, 79.
- Dehaene, S., Changeux, J.-P., Naccache, L., Sackur, J., & Sergent, C.
  (2006). Conscious, preconscious, and subliminal processing. *PLoS Biology*.
- Dehaene, S., Changeux, J.-P., & Naccache, L. (2011). The global neuronal
  workspace model of conscious access. *Neuron*, 69.
- Mashour, G. A., Ro, T., Hudetz, A. G., & Cafarelli, M. (2020). Conscious
  processing and the global neuronal workspace hypothesis. *Neuron*, 105.
- Tononi, G. (2004). An information integration theory of consciousness.
  *BMC Neuroscience*, 5.
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to
  the mechanisms of consciousness: Integrated Information Theory 3.0.
  *PLoS Computational Biology*, 10(5):e1003588.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588
- Albantakis, L., et al. (2023). Integrated information theory (IIT) 4.0.
  *PLoS Computational Biology*.
- Lamme, V. A. F. (2006). Towards a true neural stance on consciousness.
  *Trends in Cognitive Sciences*, 10(5).
- Lau, H., & Rosenthal, D. (2011). Empirical support for higher-order theories
  of conscious awareness. *Trends in Cognitive Sciences*, 15(8).
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain.*
  Oxford UP.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience*, 11.
- Friston, K., et al. (2017). Active inference: a process theory. *Neural
  Networks*. https://activeinference.github.io/papers/process_theory.pdf
- Ferrante, O., et al. (Cogitate Consortium) (2025). Adversarial testing of
  global neuronal workspace and integrated information theories of
  consciousness. *Nature*, 642. doi:10.1038/s41586-025-08888-1
  https://www.nature.com/articles/s41586-025-08888-1.pdf

**AI consciousness assessment**
- Butlin, P., Long, R., et al. (2023). Consciousness in artificial
  intelligence: Insights from the science of consciousness.
  arXiv:2308.08708. https://arxiv.org/abs/2308.08708
- Chalmers, D. J. (2023). Could a large language model be conscious?
  arXiv:2303.07103. https://arxiv.org/abs/2303.07103
- Shanahan, M. (2024). Talking about large language models. arXiv:2412.10291.
  https://arxiv.org/abs/2412.10291
- Goldstein, S., & Kirk-Giannini, C. D. (2024). Language agents and
  phenomenal consciousness. [Cited via secondary reporting; full text
  unverified — locate before relying on exact wording.]
- Chandaria, S., et al. (2025). From cacophony to hierarchy: a principled
  framework for assessing AI consciousness. arXiv:2609.35618.
  https://arxiv.org/abs/2609.35618

**Metacognition and uncertainty**
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition.
  *Frontiers in Human Neuroscience*, 8:443.
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach
  for estimating metacognitive sensitivity. *Consciousness and Cognition*, 21.
- Fleming, S. M. Metacognition and confidence: a review and synthesis.
  *Annual Review of Psychology*. https://discovery.ucl.ac.uk/id/eprint/10181587/
- Kadavath, S., et al. (2022). Language models (mostly) know what they know.
  arXiv:2207.05221.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: linguistic
  invariances for uncertainty estimation in NLG. *ICML*.
  arXiv:2302.09664.
- Lin, S., Hilton, J., & Evans, O. (2022). Teaching models to express their
  uncertainty in words. *TMLR*. arXiv:2205.14334.
- Tian, K., et al. (2023). Just ask for calibration: strategies for eliciting
  calibrated confidence scores from language models. *ICML*. arXiv:2305.14975.
- Geng, J., et al. (2024). A survey of uncertainty estimation in LLMs.
  arXiv:2410.15326.

**Cognitive architectures and agents**
- Laird, J. E. (2012). *The Soar Cognitive Architecture.* MIT Press.
- Laird, J. E., Lebiere, C., & Rosenbloom, P. S. (2017). A standard model
  of the mind. *AI Magazine*, 38(4).
- Anderson, J. R., et al. (2004). An integrated theory of the mind.
  *Psychological Review*, 111.
- Sun, R. (2007). The motivational and metacognitive control in CLARION.
  In *Modeling High-Level Cognitive Processes*.
- Franklin, S., et al. LIDA model papers. https://ccrg.cs.memphis.edu/
- Kotseruba, I., & Tsotsos, J. K. (2020). 40 years of cognitive
  architectures. *Artificial Intelligence Review*, 53:17–94.
- Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023).
  Cognitive architectures for language agents. arXiv:2309.02427.
  https://arxiv.org/abs/2309.02427
- Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in
  language models. *ICLR*. arXiv:2210.03629.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal
  reinforcement learning. *NeurIPS*. arXiv:2303.11366.
- Bengio, Y. (2017). The consciousness prior. arXiv:1709.08568.
  https://arxiv.org/abs/1709.08568
- VanRullen, R., & Kanai, R. (2021). Deep learning and the global
  workspace theory. *Trends in Neurosciences*, 44.
- Juliani, A., Kanai, R., & Sasai, S. (2021). The Perceiver architecture
  is a functional global workspace. *Proc. Cognitive Science Society*, 44.
- SEP, "Higher-Order Theories of Consciousness":
  https://plato.stanford.edu/archives/sum2020/entries/consciousness-higher/
