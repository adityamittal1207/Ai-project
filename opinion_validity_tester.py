from alternate_approach import AILegislationAnalyzer
import csv

analyzer = AILegislationAnalyzer([], "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")

viewpoints = []

with open("Viewpoints.csv", mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                viewpoints.append(row['Point'])

# print(viewpoints)

predicted_opinions = analyzer.classify_viewpoints(viewpoints)

# {'Pros': ['Existing expertise in various agencies can regulate generative AI, avoiding the need for a new AI-specific regulatory body.', 'In natural sciences, generative AI has the potential for radical breakthroughs by revealing complex underlying patterns.'], 'Cons': ['Engineers may feel terrified by the non-deterministic and high-stakes nature of generative AI, which is different from the deterministic nature of their usual work.', 'In social sciences and arts, generative AI may reinforce continuity and conformity, potentially leading to societal rigidity.'], 'Neutral': ['Generative AI is a general-purpose technology like the internet.', 'A "middle distance" view of AI is needed, focusing on the general, policy-relevant characteristics of the technology.', 'Generative AI functions as a "tradition machine," extending existing patterns rather than creating entirely new ones.', 'The existential challenge of generative AI lies more in how it reflects and amplifies human desires, rather than posing an existential threat.']}

with open("Viewpoints.csv", mode='r', encoding='utf-8') as csvfile:
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