import pandas as pd
import numpy as np

# Read the Excel file
file_name = "example_1.xlsx"  
data = pd.read_excel(file_name, header=None) 

# Extract specific variables and convert them into matrices
e = data.iloc[:4, [0]].values  
AT = data.iloc[4:, [0]].values  
BT = data.iloc[4:, 1:5].values  
b = data.iloc[:4, 5:11].values  
G = data.iloc[4:, 5:11].values  
Empty = data.iloc[:4, 1:5].values  

# For 2T2R:
# Modify matrix `e`:
e = np.hstack((e, np.zeros((e.shape[0], 1))))  
# Modify matrix `AT`: 
AT = np.hstack((AT, np.zeros((AT.shape[0], 1))))  
# Modify matrix `BT`: 
BT_new = []
for col in BT.T:  
    zero_col = np.zeros_like(col).reshape(-1, 1)  # Add a column of zeros
    BT_new.append(np.hstack((zero_col, col.reshape(-1, 1))))  

BT = np.hstack(BT_new)  
# Modify matrix `Empty`: 
Empty_new = []
for col in Empty.T:  
    zero_col = np.zeros_like(col).reshape(-1, 1)  # Add a column of zeros
    Empty_new.append(np.hstack((zero_col, col.reshape(-1, 1))))  

Empty = np.hstack(Empty_new)  

# Create the matrix `array_2T2R`
array_2T2R = np.vstack((np.hstack((e, Empty, b)), np.hstack((AT, BT, G))))  

# Replace coefficients in `array_2T2R` with resistance values
memristor_array = np.where(array_2T2R == 0, 100000, 10000 / array_2T2R)

# Write `memristor_array` to a txt file:
# Generate the content string
content1 = ""
content2 = ""
for i in range(memristor_array.shape[0]):
    for j in range(0, memristor_array.shape[1]-1, 2):
        content1 += f"parameter real Rl_{i}_{j} = {memristor_array[i][j]},  Rr_{i}_{j+1} = {memristor_array[i][j+1]};\n"   

count = 0  
for i in range(memristor_array.shape[0]):
    ja = 0
    for j in range(0, memristor_array.shape[1]-1, 2):
        content2 += f"RRAM_2T2R #(.Rl(Rl_{i}_{j}), .Rr(Rr_{i}_{j+1})) core_{count} (.WLl(WLl[{i}]), .WLr(WLr[{i}]), .BLl(BLl[{ja}]), .BLr(BLr[{ja}]), .SL(SL[{i}]));\n"
        ja = ja + 1
        count = count + 1

# Write to the txt file
with open("Memristor_array_verilogA.txt", "w") as file:
    file.write(content1)
    file.write(content2)

# Define the list of file names to save
file_names = [
    "e.xlsx",
    "AT.xlsx",
    "BT.xlsx",
    "b.xlsx",
    "G.xlsx",
    "Empty.xlsx",
    "array_2T2R.xlsx",
    "memristor_array.xlsx"
]

# Define the list of matrix variables
matrices = [e, AT, BT, b, G, Empty, array_2T2R, memristor_array]

# Save each variable to a separate Excel file
for file_name, matrix in zip(file_names, matrices):
    pd.DataFrame(matrix).to_excel(file_name, index=False, header=False)

print("All variables have been saved to separate Excel files.")