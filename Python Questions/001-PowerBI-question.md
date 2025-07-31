Question : 
### **DAX Measures:**  
1. Calculate the **total attrition rate** (percentage of employees with `Attrition = "Yes"`).  
2. Create a measure to find the **average `DistanceFromHome`** for employees working in the **"Sales" department**.  
3. Measure the **percentage of employees who work overtime (`OverTime = "Yes"`)** per `JobRole`.  
4. Calculate the **median `YearsAtCompany`** for employees with **"High" JobSatisfaction (Rating ≥ 4)**.  
5. Compute the **monthly income growth rate** (using `PercentSalaryHike`) for employees promoted in the last 2 years (`YearsSinceLastPromotion ≤ 2`).  

---

### **DAX Calculated Columns:**  
6. Create a column **`TenureCategory`**:  
   - **"New"** if `YearsAtCompany` < 2  
   - **"Mid"** if 2-5 years  
   - **"Long"** if >5 years.  
7. Add a column **`TravelFrequencyScore`**:  
   - **3** if `BusinessTravel = "Frequently"`  
   - **2** if "Travel_Rarely"  
   - **1** if "Non-Travel".  
8. Flag employees eligible for **retention programs** (`RetentionFlag`):  
   - **"Yes"** if `YearsSinceLastPromotion` > 3 AND `JobSatisfaction` < 3.  
9. Categorize `WorkLifeBalance` into **"Good" (≥3) or "Poor" (<3)** in a new column.  
10. Calculate **`HourlyToMonthlyRatio`**: `(HourlyRate * 8 * 22) / MonthlyIncome`.  

---

### **Report & Visualization Tasks:**  
11. Design a **bar chart** showing **attrition count by `JobRole`**, filtered by `MaritalStatus`.  
12. Create a **scatter plot** comparing `Age` (X-axis) and `MonthlyIncome` (Y-axis), colored by `Attrition`.  
13. Build a **slicer-based dashboard** where users can filter by `Department`, `EducationField`, and `SalarySlab`.  
14. Visualize the **distribution of `YearsSinceLastPromotion`** using a histogram.  
15. Add a **card visual** displaying the **% of employees with `StockOptionLevel > 0`**.  

---

### **Advanced DAX Challenges:**  
16. Write a measure to calculate **rolling average `MonthlyIncome`** for the last 3 years (`YearsAtCompany`).  
17. Create a **rank of departments** by `TotalWorkingYears` (highest to lowest).  
18. Measure the **correlation between `JobInvolvement` and `PerformanceRating`**.  
19. Use **SWITCH()** to categorize `EnvironmentSatisfaction` into text labels (e.g., "Low", "Medium", "High").  
20. Calculate **employee turnover risk**:  
   - High risk if `YearsAtCompany` < 2 AND `JobSatisfaction` < 2.  

---

### **Bonus:**  
- **Time Intelligence**: Add a hypothetical `HireDate` column and analyze attrition trends by year.  
- **What-If Analysis**: Model how a **10% salary hike** impacts attrition rates.  