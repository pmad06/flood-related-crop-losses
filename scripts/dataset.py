import os
import csv

input_folder = r"C:\Users\Prana\Downloads"  # change this to your folder path
output_file = r"C:\Users\Prana\Downloads\florida_flood.csv"  # output goes in same folder

headers = ["Year","State_Code","State_Abbr","County_Code","County_Name",
           "Commodity_Code","Commodity_Name","Insurance_Plan","Stage_Code",
           "Col_1","Col_2","Cause_Code","Cause_Name","Month_Code","Month_Name",
           "Col_3","Policies_Earning_Premium","Policies_Indemnified",
           "Net_Planted_Acres","Net_Endorsed_Acres","Liability","Total_Premium",
           "Producer_Premium","Subsidy","State_Private_Subsidy",
           "Additional_Subsidy","EFA_Premium_Discount","Net_Determined_Acres",
           "Indemnity","Loss_Ratio"]

flood_keywords = ["flood", "flash flood", "excess moisture"]

rows_found = 0

with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(headers)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    cols = line.strip().split('|')
                    if len(cols) < 13:
                        continue
                    state = cols[2].strip()
                    cause = cols[12].strip().lower()
                    if state == "FL" and any(k in cause for k in flood_keywords):
                        writer.writerow(cols)
                        rows_found += 1

print(f"Done! {rows_found} rows saved to florida_flood.csv")