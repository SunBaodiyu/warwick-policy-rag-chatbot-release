- Status: Frozen final test set
- Freeze date: 2026-08-04
- Total questions: 30
- Answerable: 24
- Unanswerable: 6
- Policies: IMP02, IMP03, IMP06, IMP07, IMP08, IMP09
- Language: English
- SHA-256: ad8a474a0f5c19f4acba9d77dafeb90357cb208e2f5fe44463b4b34f60f4bd28
- Statement: This test set was frozen before retrieval or generation evaluation. It must not be used for further system tuning.

# Final Test Set Draft Audit

This audit covers `data/evaluation/final_questions_draft.json`. The draft was
written from the six policy HTML files and compared manually with all 16
development questions in `data/evaluation/questions.json`. No retrieval or
generation system was run while designing the questions.

For each unanswerable question, all six policy files were searched for the
requested fact and close variants. A related rule is not treated as an answer
when the requested value, authority, format, fee, period, or sanction is not
stated.

## IMP02

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP02-Q01 | IMP02 | direct | 7 | What additional risks must be considered when using AI and publishing its output? | `potential exposure to legal penalties, financial forfeiture, and material loss` | The answer reproduces the three listed risks and their link to non-compliance with industry laws and regulations. | No. The development questions cover user responsibility, restricted data, procurement and an unspecified approved model, not publication risks. |
| FINAL-IMP02-Q02 | IMP02 | paraphrase | 6.2 | May AI be used to portray a living or deceased person as holding an opinion that is not genuinely theirs? | `AI must not be used ... to misrepresent the opinions of any person living or deceased` | The answer preserves the prohibition, the living-or-deceased scope and the false attribution condition. | No. No development question concerns misrepresentation of a person's opinions. |
| FINAL-IMP02-Q03 | IMP02 | scenario | 5 | A proposed AI deployment would create unfair or inequitable conditions for some participants. Is this permitted? | `AI must not be used to create unfair or inequitable conditions for users and participants` | The answer applies the express prohibition without adding assumptions about the deployment. | Same broad section as a development question, but tests a different clause or fact; no semantic duplication. |
| FINAL-IMP02-Q04 | IMP02 | scope_or_condition | 5 | When may information documenting the use of AI in a process be withheld, and what requirement still applies? | `Where disclosing the use of AI will prejudice the interests of the University, this information can be withheld but the requirements of UK GDPR will still need to be met.` | The answer preserves both the prejudice condition and the continuing legal requirement. | Same broad section as a development question, but tests a different clause or fact; no semantic duplication. |
| FINAL-IMP02-Q05 | IMP02 | unanswerable |  | What minimum confidence score must AI-generated material achieve before it may be published? | **not stated**; cross-policy searches for confidence, score, threshold and publication requirements found risks and quality controls but no numeric confidence threshold. | The answer refuses to invent a publication threshold absent from all six policies. | No. The development unanswerable asks for an approved model or vendor, a different missing fact. |

## IMP03

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP03-Q01 | IMP03 | direct | 6.2 | Who grants authorisation to manage University user accounts? | `granted by the Chief Digital Officer or their designate` | The answer names exactly the authority stated in the section. | No. The development questions concern accounts as digital identity and access removal for temporary leavers. |
| FINAL-IMP03-Q02 | IMP03 | paraphrase | 7.10 | How are access-permission review frequencies determined for systems generally, privileged accounts and high-risk systems? | `at least once a year`, privileged accounts `at least once a month`, and high-risk reviews `as frequent as possible` | The answer preserves the general minimum, privileged-account minimum and risk-based condition. | No. Development timing questions concern security scanning and penetration testing, not access-permission reviews. |
| FINAL-IMP03-Q03 | IMP03 | scenario | 9.4 | Monitoring detects unusual activity on a service account. What action is required? | `monitored continuously for unusual activity, and any anomalies must be investigated promptly` | The scenario directly triggers the monitoring and prompt-investigation rule. | No. The development account scenario concerns a temporary member leaving the University. |
| FINAL-IMP03-Q04 | IMP03 | scope_or_condition | 8.1 | When may a system administrator use a privileged account instead of their ordinary user account? | `specific tasks which require special privileges`; ordinary user accounts are required `at all other times` | The answer keeps both the narrow permitted condition and the default account requirement. | No. The development elevated-privilege question asks about authentication, not when a privileged account may be used. |
| FINAL-IMP03-Q05 | IMP03 | unanswerable |  | What exact naming convention must be used for service-account identifiers? | **not stated**; all six policies were checked for naming convention, identifier format, prefix and suffix. The only identifier-related rule is uniqueness. | The answer distinguishes the stated identifier property from the absent naming format without implying that service accounts have no other controls. | No. The development set does not ask about service-account identifiers. |

## IMP06

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP06-Q01 | IMP06 | direct | 3.3 | What tasks are system administrators designated to perform, and what additional responsibilities may they have? | `management, maintenance and support tasks`; additional responsibilities may include `design, deployment, development and provisioning` | The answer preserves the distinction between designated tasks and possible additional duties. | No. The development questions concern scanning, penetration testing and incident response. |
| FINAL-IMP06-Q02 | IMP06 | paraphrase | 3.1 | What kinds of digital services count as systems, and are individual hardware assets included? | `multi-user service or application`; examples include platforms, databases, cloud services and networked applications; `distinct from individual hardware assets` | The answer paraphrases the definition and its explicit exclusion without expanding it. | No. The development set does not ask for the meaning of a system. |
| FINAL-IMP06-Q03 | IMP06 | scenario | 6.1 | A department plans to migrate and upgrade a shared system. What process, review and approval are required? | `Manage systems changes in accordance with departmental change management processes`; `migrations, upgrades and changes are reviewed against current business continuity planning`; `All system changes must be approved by the System Owner or their delegate.` | The answer includes all three requirements from one policy clause. | Same broad section as a development question, but tests a different clause or fact; no semantic duplication. |
| FINAL-IMP06-Q04 | IMP06 | scope_or_condition | 6.1 | For which systems are disaster recovery plans mandatory, and what must owners do with those plans? | `Disaster recovery plans must be in place for priority systems`; plans must be written, updated, implemented and regularly tested. | The answer does not widen the requirement from priority systems to every system and retains all plan duties. | Same broad section as a development question, but tests a different clause or fact; no semantic duplication. |
| FINAL-IMP06-Q05 | IMP06 | unanswerable |  | What exact retention period applies to system access and activity logs? | **not stated**; all six policies were searched for log retention and time-unit variants. IMP07 §5.1 says personal data must not be kept longer than necessary, and IMP07 §14.1 says processing-activity records must follow the Records Retention Schedule. Logging is required, and a separate 31-day rule covers physical ID-card reader data, but none of these provisions gives an exact retention period for system logs. | The answer distinguishes related general retention requirements from the absent system-log period. | No. The development unanswerable asks for an incident-reporting deadline, not log retention. |

## IMP07

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP07-Q01 | IMP07 | direct | 4.3 | What responsibilities does the Data Protection Officer have under the policy? | `advising the University on all aspects of its compliance with data protection law`; `acting as an available point of contact with the ICO on data protection matters`; `acting as an available point of contact for complaints, and in respect of claims, from data subjects`; `monitoring our compliance with applicable data protection law, taking into account our overall risk profile, and reporting to the Audit and Risk Committee.` | The answer condenses only the responsibilities listed in section 4.3. | No. Development questions concern processing principles and privacy by design. |
| FINAL-IMP07-Q02 | IMP07 | paraphrase | 6.1 | How does a privacy notice differ from a policy governing personal-data handling? | A privacy notice is `a separate written statement informing data subjects about how their personal data is processed`; this policy is not one. | The answer preserves the explicit distinction and purpose of a privacy notice. | No. Privacy notices are absent from the development set. |
| FINAL-IMP07-Q03 | IMP07 | scenario | 15.2 | A department receives a request from an individual to exercise a statutory personal-data right. What must it do, and by when? | Act `without undue delay`, notify the central team `as soon as possible`, and meet a `one month statutory deadline`. | The answer preserves the departmental action, notification urgency and statutory deadline. | No. The development data-protection scenario concerns system design, not rights requests. |
| FINAL-IMP07-Q04 | IMP07 | scope_or_condition | 9.2 | How often must international-transfer safeguards be re-evaluated, and what event requires an additional review? | Safeguards are subject to ongoing review, re-evaluated `no less frequently than every three years`, and reviewed after a substantive legal, regulatory or operational change. | The answer retains the recurring minimum and the event-triggered condition without turning ongoing review into continuous monitoring. | No. International transfers are absent from the development set. |
| FINAL-IMP07-Q05 | IMP07 | unanswerable |  | What fee must an individual pay to exercise the right of access to their personal data? | **not stated**; all six policies were searched for fees or charges associated with data-subject rights and access requests. | The answer states only that the policy set gives no fee, without importing external legal knowledge. | No. The development unanswerables concern an AI vendor and incident-reporting hours. |

## IMP08

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP08-Q01 | IMP08 | direct | 7.1 | For how long is usage data from each ID card reader normally retained? | `retained for 31 days as standard practice or longer for investigation purposes in response to incidents` | The answer preserves both the standard period and the investigation exception. | No. The development access-control questions concern technical controls and enhanced authentication. |
| FINAL-IMP08-Q02 | IMP08 | paraphrase | 2.2 | Do University-wide access rules remove the need to follow extra controls imposed by a department or team? | Individuals must `adhere to any additional controls where required` by departments, teams, functions or business units. | The answer correctly rejects the premise that central rules replace local controls. | No. Additional local controls are not tested in the development set. |
| FINAL-IMP08-Q03 | IMP08 | scenario | 7.1 | A team films in a University room where sensitive information may be visible. What must happen before the footage is distributed? | `content must be reviewed by a University member prior to any distribution` to ensure sensitive content was not captured | The answer retains who reviews, when review occurs and its purpose. | No. The development scenario concerns authentication for elevated privileges. |
| FINAL-IMP08-Q04 | IMP08 | scope_or_condition | 6.1 | Which University-managed computers are exempt from the mandatory five-minute screen lock, and when must devices be locked? | Five-minute screen lock applies except to `Student devices and those within teaching work areas`; devices must be locked when out of sight. | The answer keeps the named exceptions and the separate out-of-sight condition. | Same broad section as a development question, but tests a different clause or fact; no semantic duplication. |
| FINAL-IMP08-Q05 | IMP08 | unanswerable |  | Which University team issues a replacement ID card after an access card is reported lost or stolen? | **not stated**; all six policies were searched for replacement cards, issuing teams, lost and stolen cards. Reporting to Community Safety is stated, but replacement authority is not. | The answer distinguishes the explicit reporting duty from the absent issuing responsibility. | No. The development set does not ask about physical access cards. |

## IMP09

| question_id | policy_id | question_type | evidence_section | Question | Supporting text or not stated | Why the answer matches | Development-set semantic overlap |
|---|---|---|---|---|---|---|---|
| FINAL-IMP09-Q01 | IMP09 | direct | 8.1 | What conditions must very occasional personal use of University information and communication facilities satisfy? | Personal use must not interfere with work, contravene policy, use excessive resources, or create costs or support burden. | The answer includes every condition listed under section 8.1. | No. The development question asks which resources the policy covers, not when personal use is allowed. |
| FINAL-IMP09-Q02 | IMP09 | paraphrase | 9.3 | May a USB storage device be attached to equipment that processes University data without advance permission? | No data-storing peripheral may be connected `unless with prior authorisation from the Information Security Risk and Compliance team`, irrespective of location. | USB storage is the policy's own example; the answer preserves the approval authority and location scope. | No. The development set has no removable-media question. |
| FINAL-IMP09-Q03 | IMP09 | scenario | 11.1 | A staff member is about to leave an ordinary logged-in workstation unattended. What must they do? | Logged-in equipment must not be `left unattended and unlocked`; purpose-built kiosks and public displays are excepted. | The answer applies the locking rule and retains the explicit exception. | No. The development scenario concerns prohibited defamatory or obscene material. |
| FINAL-IMP09-Q04 | IMP09 | scope_or_condition | 9.2 | Under what condition may a personally owned laptop or tablet be connected to the University wireless network? | Personal equipment may connect to wireless `provided they are compliant with all relevant policies`. | The answer states the only condition given and does not extend it to ordinary wired ports. | No. Personal wireless equipment is not covered by the development set. |
| FINAL-IMP09-Q05 | IMP09 | unanswerable |  | What specific disciplinary penalty is automatically imposed for a first breach involving University digital services? | **not stated**; all six policies were checked for first breach, first offence, automatic penalty and sanction. They state that certain significant-risk breaches enter applicable procedures, not that a fixed first-breach penalty applies. | The answer distinguishes a conditional disciplinary process from an automatic sanction. | No. The development unacceptable-use scenario asks whether conduct is permitted, not what first-breach penalty applies. |

## Distribution and overlap conclusion

- Each policy contributes exactly one `direct`, `paraphrase`, `scenario`,
  `scope_or_condition`, and `unanswerable` question.
- The 24 answerable questions use a single identified policy section each.
- The six unanswerable questions have empty `evidence_section` values and were
  checked across all six source policies.
- No question repeats or closely paraphrases a development-set question.
