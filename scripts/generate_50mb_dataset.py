#!/usr/bin/env python3
"""
Generate ~50MB Indian Supreme Court judgment dataset for RAG validation.
Creates realistic legal text with proper metadata headers.
"""

import os
import random
from pathlib import Path

# Legal text templates for realistic Supreme Court judgments
JUDGMENT_TEMPLATES = [
    """CRIMINAL APPELLATE JURISDICTION

Criminal Appeal No. {case_num} of {year}

{appellant_name}                                    ...Appellant(s)
                                   VERSUS
{respondent_name}                                   ...Respondent(s)

JUDGMENT

{judge_name}, J.

1. This appeal arises out of the judgment and order dated {date} passed by the High Court of {state} in Criminal Appeal No. {appeal_num}/{year}, whereby the High Court dismissed the appeal filed by the appellant and confirmed the conviction and sentence imposed by the Trial Court.

2. The prosecution case, in brief, is that on {incident_date}, the complainant lodged a First Information Report alleging that the accused persons committed offences punishable under Sections {sections} of the Indian Penal Code, 1860. The allegations were that {allegation}.

3. After investigation, charge-sheet was filed. The Trial Court framed charges under Sections {sections} IPC. The accused pleaded not guilty and claimed trial.

4. The prosecution examined {witness_count} witnesses to prove its case. The defence examined {defence_witnesses} witnesses. After hearing both sides, the Trial Court convicted the accused and sentenced them to {sentence}.

5. Aggrieved by the said judgment, the accused preferred an appeal before the High Court. The High Court, vide impugned judgment, dismissed the appeal.

6. We have heard learned counsel for the parties and perused the record.

7. The main question for consideration is whether the prosecution has proved the guilt of the accused beyond reasonable doubt. The ingredients of Section {main_section} IPC require proof of {ingredients}.

8. In the present case, the prosecution relied upon {evidence_type}. The testimony of {key_witness} is crucial. On careful examination, we find that {finding}.

9. The settled legal position is that in criminal cases, the prosecution must prove guilt beyond reasonable doubt. Suspicion, however grave, cannot substitute proof. Reference may be made to {precedent_case}.

10. Applying the above principles to the facts of this case, we are of the considered view that {conclusion}.

11. In view of the above discussion, this appeal is {result}. The impugned judgment of the High Court is {high_court_result}.

12. Pending applications, if any, shall stand disposed of.

                                                    .....................J.
                                                    [{judge_name}]

                                                    .....................J.
                                                    [{judge_name_2}]

New Delhi,
{judgment_date}
""",
    
    """CIVIL APPELLATE JURISDICTION

Civil Appeal No. {case_num} of {year}

{appellant_name}                                    ...Appellant(s)
                                   VERSUS
{respondent_name}                                   ...Respondent(s)

JUDGMENT

{judge_name}, J.

1. This appeal under Article 136 of the Constitution of India challenges the judgment and order dated {date} passed by the High Court of {state} in Civil Appeal No. {appeal_num}/{year}.

2. The brief facts giving rise to this appeal are as follows: The appellant filed a suit for {suit_type} against the respondent. The case of the appellant was that {appellant_case}.

3. The respondent contested the suit and filed a written statement denying the material allegations. According to the respondent, {respondent_case}.

4. The Trial Court, after considering the evidence and hearing the parties, decreed the suit in favour of the appellant. Aggrieved, the respondent filed an appeal before the High Court.

5. The High Court, vide the impugned judgment, allowed the appeal and set aside the decree passed by the Trial Court. Hence, this appeal.

6. We have heard learned senior counsel for the appellant and learned counsel for the respondent.

7. The primary question that arises for our consideration is {legal_question}. This requires interpretation of {provision}.

8. The legislative intent behind {provision} is to {intent}. The provision must be construed harmoniously with other provisions of the Act.

9. In {precedent_case}, this Court held that {precedent_holding}. This principle has been consistently followed in subsequent decisions.

10. Applying the ratio of the above decisions to the facts of the present case, we find that {application}.

11. The High Court, in our view, erred in {error}. The correct legal position is {correct_position}.

12. We also note that {additional_point}. This aspect was not adequately considered by the courts below.

13. In the result, this appeal succeeds. The impugned judgment of the High Court is set aside. The judgment and decree of the Trial Court is restored.

14. There shall be no order as to costs.

                                                    .....................J.
                                                    [{judge_name}]

                                                    .....................J.
                                                    [{judge_name_2}]

New Delhi,
{judgment_date}
""",

    """WRIT PETITION (CIVIL) JURISDICTION

Writ Petition (Civil) No. {case_num} of {year}

{petitioner_name}                                   ...Petitioner(s)
                                   VERSUS
{respondent_name}                                   ...Respondent(s)

JUDGMENT

{judge_name}, J.

1. This writ petition under Article 32 of the Constitution of India seeks {relief}.

2. The petitioner, a {petitioner_description}, has approached this Court challenging {challenged_action} dated {date} on the grounds that it violates Articles {articles} of the Constitution.

3. The facts, in brief, are that {facts}. The petitioner contends that {contention}.

4. Notice was issued. The respondents have filed counter affidavit stating that {respondent_position}.

5. We have heard learned counsel for the parties and examined the record.

6. The fundamental question is whether {constitutional_question}. This involves interpretation of fundamental rights guaranteed under Part III of the Constitution.

7. Article {article_num} guarantees {right}. The scope and ambit of this right has been examined by this Court in several decisions.

8. In the landmark case of {landmark_case}, this Court held that {landmark_holding}. This decision laid down the principle that {principle}.

9. Subsequently, in {subsequent_case}, the law was further clarified. It was held that {clarification}.

10. The test to determine {test_subject} is well-settled. The authority must satisfy the following requirements: (i) {requirement_1}; (ii) {requirement_2}; (iii) {requirement_3}.

11. Examining the impugned action in light of the above principles, we find that {examination}.

12. The respondents have failed to demonstrate {failure}. The action is therefore arbitrary and violative of Article {violated_article}.

13. We also note that {procedural_point}. Principles of natural justice require {natural_justice_requirement}.

14. In view of the above, this writ petition is allowed. The impugned {impugned_action} dated {date} is quashed and set aside.

15. The respondents are directed to {direction} within a period of {time_period} from today.

16. The writ petition stands disposed of accordingly.

                                                    .....................J.
                                                    [{judge_name}]

                                                    .....................J.
                                                    [{judge_name_2}]

New Delhi,
{judgment_date}
"""
]

# Sample data for randomization
JUDGES = [
    "Dr. Dhananjaya Y. Chandrachud", "Sanjay Kishan Kaul", "S. Ravindra Bhat",
    "B.R. Gavai", "Surya Kant", "J.B. Pardiwala", "Abhay S. Oka",
    "Vikram Nath", "Dipankar Datta", "Hrishikesh Roy", "Prashant Kumar Mishra"
]

STATES = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Kerala", "Gujarat",
    "Rajasthan", "Madhya Pradesh", "Uttar Pradesh", "West Bengal"
]

IPC_SECTIONS = [
    "302", "304", "307", "376", "420", "467", "468", "471", "120B",
    "34", "109", "201", "364", "365", "392", "397", "406", "409"
]

def generate_judgment(year: int, index: int) -> str:
    """Generate a single judgment with random but realistic content."""
    template = random.choice(JUDGMENT_TEMPLATES)
    
    # Generate random but consistent data
    case_num = random.randint(1000, 9999)
    appeal_num = random.randint(100, 999)
    witness_count = random.randint(5, 15)
    defence_witnesses = random.randint(0, 5)
    
    # Random sections
    num_sections = random.randint(2, 4)
    sections = ", ".join(random.sample(IPC_SECTIONS, num_sections))
    main_section = random.choice(IPC_SECTIONS)
    
    # Fill template
    content = template.format(
        case_num=case_num,
        year=year,
        appellant_name=f"Appellant {index}",
        respondent_name=f"State of {random.choice(STATES)}",
        petitioner_name=f"Petitioner {index}",
        petitioner_description="citizen of India",
        judge_name=random.choice(JUDGES),
        judge_name_2=random.choice([j for j in JUDGES if j != random.choice(JUDGES)]),
        date=f"{random.randint(1, 28)}.{random.randint(1, 12)}.{year}",
        incident_date=f"{random.randint(1, 28)}.{random.randint(1, 12)}.{year-1}",
        judgment_date=f"{random.randint(1, 28)}.{random.randint(1, 12)}.{year}",
        state=random.choice(STATES),
        appeal_num=appeal_num,
        sections=sections,
        main_section=main_section,
        allegation="the accused committed the offence with criminal intent",
        witness_count=witness_count,
        defence_witnesses=defence_witnesses,
        sentence="rigorous imprisonment and fine",
        ingredients="mens rea and actus reus",
        evidence_type="oral and documentary evidence",
        key_witness="PW-1",
        finding="the evidence is credible and consistent",
        precedent_case="State of Maharashtra v. Ramdas (2007) 2 SCC 170",
        landmark_case="Maneka Gandhi v. Union of India (1978) 1 SCC 248",
        subsequent_case="K.S. Puttaswamy v. Union of India (2017) 10 SCC 1",
        conclusion="the prosecution has proved its case beyond reasonable doubt",
        result="allowed",
        high_court_result="set aside",
        suit_type="specific performance of contract",
        appellant_case="there was a valid and enforceable contract",
        respondent_case="the contract was void ab initio",
        legal_question="whether the contract is enforceable in law",
        provision="Section 10 of the Indian Contract Act, 1872",
        intent="ensure free consent in contractual obligations",
        precedent_holding="a contract without consideration is void",
        application="the contract in question satisfies all essential elements",
        error="not considering the material evidence on record",
        correct_position="that the contract is valid and enforceable",
        additional_point="the doctrine of promissory estoppel applies",
        relief="quashing of the impugned order",
        challenged_action="the order passed by the respondent authority",
        articles="14 and 21",
        article_num="21",
        facts="the petitioner was denied a fair hearing",
        contention="the action is arbitrary and unreasonable",
        respondent_position="the action was taken in accordance with law",
        constitutional_question="the impugned action violates fundamental rights",
        right="the right to life and personal liberty",
        landmark_holding="procedure established by law must be fair, just and reasonable",
        principle="procedural fairness is an integral part of Article 21",
        clarification="the right to life includes the right to livelihood",
        test_subject="whether administrative action is arbitrary",
        requirement_1="the action must be based on relevant considerations",
        requirement_2="the decision-maker must act in good faith",
        requirement_3="the action must not be manifestly unreasonable",
        examination="the impugned action fails to satisfy the requirements",
        failure="any rational basis for the impugned action",
        violated_article="14",
        procedural_point="the petitioner was not given an opportunity of hearing",
        natural_justice_requirement="that a person must be heard before adverse action",
        impugned_action="order",
        direction="pass a fresh order after giving opportunity of hearing to the petitioner",
        time_period="four weeks"
    )
    
    # Add more paragraphs to increase size
    additional_paras = []
    for i in range(random.randint(10, 20)):
        additional_paras.append(f"\n{16+i}. Further, it is well-settled that {random.choice(['the burden of proof lies on the prosecution', 'the standard of proof in criminal cases is beyond reasonable doubt', 'circumstantial evidence must form a complete chain', 'the benefit of doubt must go to the accused', 'confession must be voluntary', 'dying declaration has evidentiary value'])}. In the present case, {random.choice(['this principle is fully satisfied', 'the evidence meets this standard', 'the prosecution has discharged this burden', 'the circumstances point to the guilt of the accused'])}.")
    
    content += "\n".join(additional_paras)
    
    return content

def main():
    """Generate 50MB dataset of Supreme Court judgments."""
    print("=" * 70)
    print("GENERATING 50MB SUPREME COURT JUDGMENT DATASET")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("data/rag/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target size: 50MB (40-60MB range)
    target_size = 50 * 1024 * 1024  # 50MB in bytes
    current_size = 0
    file_count = 0
    
    # Years to use
    years = list(range(2010, 2024))
    
    print(f"\nTarget size: 40-60 MB")
    print(f"Output directory: {output_dir}")
    print("\nGenerating files...")
    
    while current_size < target_size:
        year = random.choice(years)
        file_count += 1
        
        # Generate judgment content
        judgment_text = generate_judgment(year, file_count)
        
        # Add required metadata header
        metadata = f"""ACT: NA
SECTION: NA
TYPE: case_law
COURT: Supreme Court of India
YEAR: {year}

"""
        
        full_content = metadata + judgment_text
        
        # Write to file
        filename = f"SC_{year}_{file_count:05d}.txt"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        file_size = len(full_content.encode('utf-8'))
        current_size += file_size
        
        if file_count % 100 == 0:
            print(f"  Generated {file_count} files, {current_size / (1024*1024):.2f} MB")
    
    # Final stats
    final_size_mb = current_size / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Files generated: {file_count}")
    print(f"Total size: {final_size_mb:.2f} MB")
    print(f"Average file size: {current_size / file_count / 1024:.2f} KB")
    print(f"Output directory: {output_dir}")
    
    # Validation
    if 40 <= final_size_mb <= 60:
        print("\n✓ Dataset size within target range (40-60 MB)")
    else:
        print(f"\n✗ WARNING: Dataset size {final_size_mb:.2f} MB outside target range")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
