#!/usr/bin/env python3
"""
Generate ~500MB Indian Supreme Court Judgments Dataset
Creates synthetic but realistic Supreme Court judgments for RAG testing
"""

import os
import random
from pathlib import Path
from typing import List

# Legal text templates for realistic content
JUDGMENT_TEMPLATES = [
    """The appellant filed a Special Leave Petition challenging the judgment and order passed by the High Court. The facts of the case are that the respondent filed a complaint alleging criminal breach of trust and cheating. The trial court convicted the appellant under relevant sections. The High Court dismissed the appeal.

After hearing learned counsel for both parties and perusing the record, we find that the prosecution has successfully established the guilt of the appellant beyond reasonable doubt. The evidence on record clearly demonstrates that the appellant committed the offence with dishonest intention.

The ingredients of the offence are: (1) entrustment of property, (2) misappropriation or conversion of such property, and (3) dishonest intention. All these elements have been proved by the prosecution through documentary and oral evidence.

The appellant's contention that there was no dishonest intention is not sustainable. The conduct of the appellant, as evident from the record, clearly shows mens rea. The appellant deliberately failed to account for the entrusted property and used it for personal benefit.

We have carefully considered the submissions made by learned counsel for the appellant regarding the quantum of sentence. However, considering the gravity of the offence and the amount involved, we find no reason to interfere with the sentence imposed by the trial court.

Accordingly, the appeal is dismissed. The conviction and sentence are upheld. The appellant shall surrender to custody within four weeks to serve the remaining sentence.""",

    """This appeal arises from the judgment of the High Court upholding the conviction of the appellant. The prosecution case is that the appellant, being a public servant, demanded and accepted illegal gratification for performing official duties.

The primary question for consideration is whether the prosecution has proved the essential ingredients of the offence beyond reasonable doubt. The ingredients are: (1) the accused must be a public servant, (2) acceptance of gratification, (3) such gratification must be other than legal remuneration, and (4) it must be for performing or forbearing to perform official acts.

The evidence of the complainant is corroborated by the testimony of independent witnesses who were present during the trap proceedings. The chemical analysis of the currency notes confirms the presence of phenolphthalein powder. The recovery of the tainted money from the appellant's possession is duly proved.

The defence contention that the money was planted is not credible. The trap proceedings were conducted in accordance with established procedure. The independent witnesses have no reason to falsely implicate the appellant.

Regarding the quantum of punishment, we note that corruption in public office strikes at the root of public administration and erodes public confidence in governance. The sentence imposed is proportionate to the gravity of the offence.

For the foregoing reasons, we find no merit in this appeal. The same is dismissed. The conviction and sentence are confirmed.""",

    """The present appeal challenges the constitutional validity of certain provisions of the statute. The petitioner contends that the impugned provisions violate fundamental rights guaranteed under Articles 14, 19, and 21 of the Constitution.

The first question is whether the classification made by the statute is reasonable and has rational nexus with the object sought to be achieved. The test of reasonable classification is well-settled. It must be founded on intelligible differentia which distinguishes persons or things grouped together from others left out, and such differentia must have rational relation to the object sought to be achieved.

Applying this test, we find that the classification is based on intelligible differentia. The statute seeks to regulate a specific class of activities which have distinct characteristics. The differentia has direct nexus with the legislative objective.

The second contention relates to alleged violation of Article 19. The right to freedom of speech and expression is not absolute. It is subject to reasonable restrictions in the interest of public order, decency, morality, and sovereignty and integrity of India.

The restrictions imposed by the impugned provisions are reasonable and proportionate. They serve a legitimate state interest and do not impose undue burden on the exercise of fundamental rights. The means adopted are rationally connected to the objective.

After detailed consideration of the submissions and precedents cited, we hold that the impugned provisions are constitutionally valid. The petition is dismissed.""",

    """This criminal appeal arises from conviction under provisions relating to criminal conspiracy and forgery. The prosecution alleged that the appellant, in conspiracy with others, fabricated documents to defraud the complainant.

The essential ingredients of criminal conspiracy are: (1) agreement between two or more persons, (2) such agreement must be for doing an illegal act or a legal act by illegal means, and (3) some overt act must be done in pursuance of the conspiracy.

The prosecution has led evidence to establish that the appellant entered into an agreement with co-accused to forge documents. The documentary evidence clearly shows that the signatures were forged and the documents were fabricated. The handwriting expert's report confirms forgery.

The appellant's defence that he was not involved in the conspiracy is belied by the evidence on record. The testimony of witnesses establishes his active participation. The recovery of incriminating documents from his possession further strengthens the prosecution case.

On the question of sentence, we note that forgery and criminal conspiracy are serious offences that undermine the integrity of legal documents and commercial transactions. The sentence imposed by the trial court is appropriate.

However, considering that the appellant is a first-time offender and has already undergone substantial custody, we are inclined to reduce the sentence. The sentence is modified to the period already undergone. The appeal is partly allowed to this extent.""",

    """The appellant challenges the order of conviction for offences under the Prevention of Corruption Act. The case involves allegations of possession of disproportionate assets by a public servant.

The prosecution must prove: (1) the accused is a public servant, (2) during the check period, the accused held the office, (3) the accused possessed pecuniary resources or property disproportionate to known sources of income, and (4) the accused is unable to satisfactorily account for such disproportionate assets.

The prosecution has filed detailed statements showing the assets held by the appellant at the beginning and end of the check period. The income from known sources has been calculated. The disproportionate assets amount to substantial percentage of the known income.

The appellant's explanation regarding the source of assets is not satisfactory. The claim that the assets were acquired from agricultural income is not supported by credible evidence. The income tax returns do not reflect such agricultural income.

The burden of proof shifts to the accused to explain the source of disproportionate assets. The explanation offered must be reasonable and probable. In the present case, the explanation is neither reasonable nor supported by evidence.

The conviction is based on credible evidence and proper appreciation of facts. We find no infirmity in the judgment of the courts below. The appeal is dismissed and the conviction is upheld."""
]

LEGAL_TOPICS = [
    "criminal breach of trust", "cheating and fraud", "corruption and bribery",
    "constitutional interpretation", "fundamental rights", "criminal conspiracy",
    "forgery and fabrication", "disproportionate assets", "money laundering",
    "criminal negligence", "defamation", "wrongful confinement", "assault",
    "criminal intimidation", "public nuisance", "criminal trespass",
    "theft and robbery", "extortion", "dishonest misappropriation",
    "breach of contract", "specific performance", "injunction",
    "property disputes", "succession and inheritance", "partition",
    "matrimonial disputes", "maintenance", "custody of children",
    "consumer protection", "service matters", "labour disputes",
    "tax evasion", "customs violations", "environmental law",
    "intellectual property", "arbitration", "contempt of court",
    "habeas corpus", "preventive detention", "bail applications",
    "sentencing guidelines", "evidence appreciation", "witness credibility"
]

def generate_judgment_text(year: int, case_num: int) -> str:
    """Generate realistic judgment text."""
    
    # Select random template
    template = random.choice(JUDGMENT_TEMPLATES)
    
    # Add case details
    topic = random.choice(LEGAL_TOPICS)
    
    header = f"""SUPREME COURT OF INDIA

Criminal Appeal No. {case_num} of {year}

Decided on: {random.randint(1, 28)}/{random.randint(1, 12)}/{year}

CORAM:
Hon'ble Justice {random.choice(['R.M. Lodha', 'A.K. Sikri', 'S.A. Bobde', 'N.V. Ramana', 'D.Y. Chandrachud'])}
Hon'ble Justice {random.choice(['Rohinton Nariman', 'Uday Lalit', 'Indu Malhotra', 'Hemant Gupta', 'Surya Kant'])}

Subject: {topic.title()}

JUDGMENT

"""
    
    # Add sections
    sections = []
    
    # Introduction
    sections.append("1. INTRODUCTION\n\n" + template[:300])
    
    # Facts
    sections.append("\n\n2. FACTS OF THE CASE\n\n" + template[300:600])
    
    # Arguments
    sections.append("\n\n3. SUBMISSIONS\n\n" + template[600:900])
    
    # Analysis
    sections.append("\n\n4. ANALYSIS AND FINDINGS\n\n" + template[900:1200])
    
    # Conclusion
    sections.append("\n\n5. CONCLUSION\n\n" + template[1200:])
    
    # Add legal citations
    citations = f"""

CITATIONS:
- AIR {year} SC {random.randint(1000, 9999)}
- ({year}) {random.randint(1, 16)} SCC {random.randint(100, 999)}
- {year} SCC Online SC {random.randint(1000, 9999)}

RESULT: Appeal {random.choice(['dismissed', 'allowed', 'partly allowed'])}
"""
    
    return header + "".join(sections) + citations


def generate_dataset(target_size_mb: float = 500, output_dir: str = "data/rag/raw"):
    """Generate Supreme Court judgments dataset.
    
    Args:
        target_size_mb: Target size in MB
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating ~{target_size_mb}MB Supreme Court judgments dataset...")
    print(f"Output directory: {output_path}")
    print()
    
    target_size_bytes = target_size_mb * 1024 * 1024
    current_size = 0
    file_count = 0
    
    years = list(range(2010, 2024))  # 2010-2023
    
    file_sizes = []
    
    while current_size < target_size_bytes:
        year = random.choice(years)
        case_num = random.randint(1000, 9999)
        
        # Generate judgment text
        judgment_text = generate_judgment_text(year, case_num)
        
        # Add required metadata header
        metadata = f"""ACT: NA
SECTION: NA
TYPE: case_law
COURT: Supreme Court of India
YEAR: {year}

"""
        
        full_text = metadata + judgment_text
        
        # Create filename
        filename = f"SC_{year}_{case_num:04d}.txt"
        filepath = output_path / filename
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        file_size = len(full_text.encode('utf-8'))
        file_sizes.append(file_size)
        current_size += file_size
        file_count += 1
        
        if file_count % 50 == 0:
            current_mb = current_size / (1024 * 1024)
            print(f"  Generated {file_count} files ({current_mb:.1f} MB / {target_size_mb} MB)")
    
    # Final stats
    final_size_mb = current_size / (1024 * 1024)
    avg_size_kb = (current_size / file_count) / 1024
    min_size_kb = min(file_sizes) / 1024
    max_size_kb = max(file_sizes) / 1024
    
    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total files:      {file_count:,}")
    print(f"Total size:       {final_size_mb:.2f} MB")
    print(f"Average file:     {avg_size_kb:.2f} KB")
    print(f"Smallest file:    {min_size_kb:.2f} KB")
    print(f"Largest file:     {max_size_kb:.2f} KB")
    print("=" * 70)
    print()
    
    # Verify size is in range
    if 400 <= final_size_mb <= 550:
        print("✓ Dataset size within target range (400-550 MB)")
    else:
        print(f"⚠ Dataset size outside target range: {final_size_mb:.2f} MB")
    
    print()
    return file_count, final_size_mb


if __name__ == "__main__":
    generate_dataset(target_size_mb=500)
