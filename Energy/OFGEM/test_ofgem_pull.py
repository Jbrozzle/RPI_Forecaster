import xlwings as xw
import pandas as pd

workstation = "MAC"

# wb = xw.Book.caller()
if workstation == "MAC":
    annex_2_wb = xw.Book("/Users/Josh/Desktop/RPI_Forecaster/Energy/OFGEM/Annex_2_-_wholesale_cost_allowance_methodology_v1.12 (input_data_removed).xlsx")
    cap_calculator_wb = xw.Book("/Users/Josh/Desktop/RPI_Forecaster/Energy/OFGEM/Default_tariff_cap_level_v1.12.xlsx")
if workstation == "PC":
    pass

elec_offset = int(annex_2_wb.sheets('3d(ii) Price data elec Q+n').range('h6').value)
gas_offset = int(annex_2_wb.sheets('3e Price data gas').range('i6').value)

elec_prices_used = (annex_2_wb.sheets("Josh Price elec Q+n").range('l1:t23').offset(elec_offset).options(pd.DataFrame).value.dropna())
gas_prices_used = annex_2_wb.sheets("Josh Price gas").range('k1:s16').offset(gas_offset).options(pd.DataFrame).value.dropna()

cap_mom = cap_calculator_wb.sheets("Cap Table").range('J5:L77').options(pd.DataFrame).value.dropna()


print(elec_prices_used)