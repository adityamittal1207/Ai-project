from alternate_approach import AILegislationAnalyzer
import csv

analyzer = AILegislationAnalyzer([], "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")

viewpoints = []

with open("Viewpoints2.csv", mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                viewpoints.append(row['Point'])

# print(viewpoints)

predicted_opinions = analyzer.classify_viewpoints(viewpoints)

# {'Pros': ['Existing expertise in various agencies can regulate generative AI, avoiding the need for a new AI-specific regulatory body.', 'In natural sciences, generative AI has the potential for radical breakthroughs by revealing complex underlying patterns.'], 'Cons': ['Engineers may feel terrified by the non-deterministic and high-stakes nature of generative AI, which is different from the deterministic nature of their usual work.', 'In social sciences and arts, generative AI may reinforce continuity and conformity, potentially leading to societal rigidity.'], 'Neutral': ['Generative AI is a general-purpose technology like the internet.', 'A "middle distance" view of AI is needed, focusing on the general, policy-relevant characteristics of the technology.', 'Generative AI functions as a "tradition machine," extending existing patterns rather than creating entirely new ones.', 'The existential challenge of generative AI lies more in how it reflects and amplifies human desires, rather than posing an existential threat.']}

with open("Viewpoints2.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    writerows = []

    for row in reader:
        predicted_opinion = None
        for opinion, points in predicted_opinions.items():
            if row['Point'] in points:
                predicted_opinion = opinion.replace("s","")
                break
        row['Predicted Opinion'] = predicted_opinion
        writerows.append(row)

with open("Viewpoints.csv", mode='w', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(writerows)