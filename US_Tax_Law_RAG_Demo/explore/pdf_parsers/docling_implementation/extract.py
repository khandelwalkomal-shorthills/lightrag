from docling.document_converter import DocumentConverter
import json

pdf_path = "your pdf path"

converter = DocumentConverter()
result = converter.convert(pdf_path)


# data=result.document.export_to_markdown()

# with open('output_data.md', 'w') as file:
#     file.write(data)

# # Save the data in .markdown format
# with open('output_data.markdown', 'w') as file:
#     file.write(data)

# # Save the data in .txt format
# with open('output_data.txt', 'w') as file:
#     file.write(data)


# import markdown
# html_content = markdown.markdown(data)

# with open("output_data.html", 'w') as file:
#     file.write(html_content)


data=result.document.export_to_dict()
with open('output_data.json', 'w') as file:
    json.dump(data, file, indent=4)

# result.document.export_to_text()
# result.document._export_to_indented_text()

# t=result.model_dump_json()

# print(type(t))
# f=json.loads(t)
# with open('output.json', 'w') as file:
#     json.dump(f, file, indent=4)
