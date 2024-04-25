<div align="center" >
  <h1>Credit Risk Modelling</h1> 
  <p>Determining the level of credit risk associated with extending credit to a borrower</p>
</div>

<div align="center" >
  <h2>Methodology</h2>
  <img src="Resources/methodology.png" alt="method"style="width:300px;height:500px;">
</div>

<div>
  <h2 align = "center">
    Summary
  </h2>  
  <p>
    The dataset contained around 40 variables and thousands of customers.The target variable was <b>Approved Flag</b>. <br> Chi-square test was performed on categorical variables to check the association with the target variable.<br> and all the categorical variables were label encoded.<br>
Three models were used -
   <ul>
  <li>Decision Tree</li>
  <li>Random Forest</li>
  <li>XGBoost</li>
</ul>    
  XGBoost gave the best result as compared to the other two
  To increase the accuracy and improve the model, hyperparameter tuning was performed using <b>grid search method </b>. 
  </p>
</div>
