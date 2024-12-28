import csv

import openpyxl

# load worksheet
xlsx_path = "IOC_10.1_vs_other_lists.xlsx"
workbook = openpyxl.load_workbook(xlsx_path)
sheet = workbook.active

def find_green_and_b_column(search_value):
    # search target value in worksheet
    for row in sheet.iter_rows():
        for cell in row:
            # print(cell.value)
            if str(cell.value).strip() == search_value.strip():
                # row index of target cell
                target_row = cell.row

                # find column B cell of the same row
                b_value = sheet[f"B{target_row}"].value

                # find first green cell
                green_value = None
                for r in range(target_row, 0, -1):
                    cell_to_check = sheet[f"B{r}"]
                    fill = cell_to_check.fill
                    if fill and fill.start_color.rgb == "FF92D050":  # green value used by IOC
                        green_value = cell_to_check.value
                        break

                return b_value, green_value

    return None, None


def process_list(file_path, export_path):
    with open(file_path, 'r') as in_file:
        with open(export_path, 'w') as out_file:
            reader = csv.reader(in_file)
            writer = csv.writer(out_file, delimiter=",")
            for row in reader:
                b_value, green_value = find_green_and_b_column(row[0])

                writer.writerow([row[0], b_value, green_value])


if __name__ == '__main__':
    # print(find_green_and_b_column('Chrysophlegma humii'))
    process_list(
        '',
        '')