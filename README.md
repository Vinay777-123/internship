# internship
# Titanic_Servived Prediction
titanic Servived Analyse
# Data Source
https://www.kaggle.com/c/titanic/data
# Module Accuress
97.3621103117506%
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4727b9c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "feb815c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "de973c41",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4a772bf8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2a1a6e1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data=pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "cec3b9ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_test_real_values=pd.read_csv('gender_submission.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1e5ea620",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data=pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fdb94351",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8503e2ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "train_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "2173d083",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "253ed6fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "140ee3c6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x7f35bf882490>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABNcAAATXCAYAAADN65MkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOzdeZxcV33n/e/v1tLVq5ZWa0FSa7FlsCXb4KfHGAcTYhPGMEYmZjGEgQTMIzIDMRmYLDNPxo5EZp6QhedlswRMIGHJBDs4AZlxHDI2ixnHYBkwWDZgWbZlyVpaLamX6q7t3vP8UXWrq6qrW11X3dXb5/161aur7j33nN89data/dO555hzTgAAAAAAAAAa5811AAAAAAAAAMBCRXINAAAAAAAAiIjkGgAAAAAAABARyTUAAAAAAAAgIpJrAAAAAAAAQEQk1wAAAAAAAICI4nMdwDlycx0AliSb6wAAAAAAAMD8wMg1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERyDQAAAAAAAIiI5BoAAAAAAAAQEck1AAAAAAAAICKSawAAAAAAAEBEJNcAAAAAAACAiEiuAQAAAAAAABGRXAMAAAAAAAAiIrkGAAAAAAAARERy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9g1d+M+xfs9IMselrXPd32Q2mLEazHMXr7Ht71zNtCdotp1p5zIpGn3j4bpKk1O8r3U8V4mIiIhoPBo3k2sAvgDgE0mS2iRJ8gP4C4DLx7hPmgocNmy8fi7MRh227WvA2kXFypeHHQca8ch15crvch6vAoftnNuKx/4ms8GMVX/P3WiObVlWIjYsU58nG5aVoywrccTbJiIaLbzWEREREdH5QJAkaaz7AAAQBOESAL8FcBGAHoQKGVRJkvTLWI+prKyUqqqqRqeDEeSKk61dHmTZQxUn27o9yEjqqxba2hX6PV7VQuO1v8lsMGPV33M3yLEd8uBrnausoEejZEjn61heV2lyGsK1jucqTRRD/RwwPj5s0/mIXxiIiEbJuKkWKknSe4IgvATgQwABAB8B2Dy2vYpNpxNQmJ6IwvS+v75Pz+j7OfK+eLdF2gYzVgM9d6PFZknAxdM4mUZEkxuvdUREREQ02Y2byTUAkCTpIQAPjXU/iIiIiIiIiIiIBmM85VwjIiIiIiIiIiKaUDi5RkRERERERERENEycXCMiIiIiIiIiIhomTq4RERERERERERENEyfXiIiIiIiIiIiIhomTa0RERERERERERMPEyTUiIiIiIiIiIqJh4uQaERERERERERHRMHFyjYiIiIiIiIiIaJg4uUZERERERERERDRMhrHuABERERHR+SgQCODo0aPK7zNnzoTBwI/nREREEw3fvYmIiIiIxsDRo0dx+1M7kJSZh66Wevz6DqCiomKsu0VERERDxMk1IiIiIqIxkpSZh5Sc6WPdDSIiIjoHzLlGREREREREREQ0TJxcIyIiIiIiIiIiGiZOrhEREREREREREQ0TJ9eIiIiIiIiIiIiGiZNrREREREREREREw8TJNSIiIiIiIiIiomHi5BoREREREREREdEwcXKNiIiIiIiIiIhomDi5RkRERERERERENEycXCMiIiIiIiIiIhomTq4RERERERERERENEyfXiIiIiIiIiIiIhomTa0RERERERERERMPEyTUiIiIiIiIiIqJh4uQaERERERERERHRMI2ryTVBEFIEQXhJEIRjgiAcFQThsrHuExERERERERERUSyGse5AhCcA7JQkaYUgCCYA1rHuEBERERERERERUSzjZnJNEIRkAAsA/DMASJLkA+Abyz4RERERERERERH1ZzwtC50G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IiIiIiIiIqJyW8rfXCMiIiIiIiIiIiorDq4RERERERERERGViINrREREREREREREJarrwbVt27ZJAPzhT6V/isa2yp8q/hSFbZU/VfwpCtsqf6r4UxS2Vf5U8acU1Y6ZP8vzh6ju1fXg2pUrV6odAtGCsK1SvWBbpXrBtkr1gm2ViIho6avrwTUiIiIiIiIiIqJqqqnBNSFEixDim0KIV4UQJ4UQ76p2TERERERERERERPk4qx1AhocAPCOl/IgQwg2gsdoBUWXousS5yTDGQlF0+L3oafVBUUS1wyqr5VhmIlp+eK6rfTxG9Y3Hj4iIqPpqZnBNCBEA8B4AnwAAKWUcQLyaMVFl6LrEM8OjuO+JlxFN6PC6FDy4ezO29XUu2YvD5VhmIlp+eK6rfTxG9Y3Hj4iIqDbU0m2h6wFMAPhHIcTPhBB/L4TwVTsoKr9zk2HzohAAogkd9z3xMs5NhqscWfksxzIT0fLDc13t4zGqbzx+REREtaGWBtecAG4B8DdSyrcBCAP4g8yNhBD3CCGOCiGOTkxMVDpGKoOxUNS8KDREEzrGp6NVisgehdrqUi0z1SeeV6lc7D7Xsa3aj/1ReVSqrfL4ERER1YZaGly7COCilPInqdffRHKwLY2U8hEp5aCUcrCtra2iAVJ5dPi98LrSm6LXpaC92VuliOxRqK0u1TJTfeJ5lcrF7nMd26r92B+VR6XaKo8fVVL3dWshhCjqp/u6tdUOm4ioImrmmWtSylEhxJtCiBullK8B2ArglWrHReXX0+rDg7s3Zz0vpKd16d4VvBzLTETLD891tY/HqL7x+FElXb74Ju786vNF7fP4b91WpmiIiGpLzQyupfwOgK+nZgo9A+A3qhwPVYCiCGzr68Rb9rwb49NRtDcv/ZmulmOZiWj54bmu9vEY1TcePyIiotpQU4NrUsqXAQxWOw6qPEUR2NDWhA1tTdUOpWKWY5mJaPnhua728RjVNx4/IiKi6qulZ64RERERERERERHVFQ6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJOLhGRERERERERERUIg6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJOLhGRERERERERERUIg6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJ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pZRo9VlmC13RCK/LgSvhGNqb55ktNBxHd6ABfasDabOFGvlZZwuVUqLZs8DZQhPJWTHXpmYLHQ1GsXYBs4XOxlUz3vHp5Ou1OWYLDceTt6Tkmy00Vzq5XusSmJiJor15brbQ0WAUnan6zzdb6NVw8ljON1uohDRnXY2qGuKqnhZHQtPS6tSYiW18OmoeuxwPWS76qcu52ipnGKTFKKL9FNVe2VaLx/oprJJtFciYLXQ6hlafG80ey2yh0zG0N+eZLRTAbNwyW2g0gWZvcrbQFY1ubLLMFmr0VTd1+NNmC52OqpiOqmhrckOTGrwuF25sa86eLbTRDV3OzRY6FY5jZZMHbnO20Dg6/R40e10YS/VJ1j6qK5CcLXQ0lN5nps0W2uxB14r8s4Va+/iFzhY6T/84L7vSqTJb2mqtzhZqzqrJ2UJr2gLruZQ3V94Ph0II3PnV54tK7PHfug218nmTalrddQREmWpmcK0UhQbXiMrIlsE1ogqx5UMgUQWwrVK9YFulesHBNaoXHFyjulczt4USERERERERERHVGw6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJOLhGRERERERERERUIg6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJOLhGRERERERERERUIg6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJOLhGRERERERERERUIg6uERERERERERERlYiDa0RERERERERERCXi4BoREREREREREVGJampwTQjxD0KIcSHEULVjISIiIiIiIiIimo+z2gFk+O8AvgzgscUkousSF66GMTkThyp1qJpELKFjXasP61Y24uLULMZCMcQ1DX6PC7MJDR1+L3pafdB1iVfHQrg2m4Cu62j1eRBTdURVDbGEjvWrfACAq+E4ICQcQkACUHUJQAIQUDUdmg7E1bk8L1ybxVgoiq6AF1LO7e92KNB1IJzQcC0cR/eKRry1oxkXgxEEI3EICAAymZ6mQdMBpwJ4XU5EVQ2qJiGlhN+bXY7hkSAmwzG0NXkxG9cwHUtglc8DVZeIWcqjS+DqbAxNHgemIxquhGPobmlEX5cfTqcCXZe4OBXG6FQM49Mx9LQ2QtUlrs7G4Pe6EVd1dK/wYjqi4lokAYeQcDuduDIT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RAjxPwG8F8AqAGMA9kkpv5Zv+8HBQXn06NGs5blmC42rOtauTJ8tNKFpaJ5nttCVPg/iqdlC46puXqyVMlvo+HQUnf7cs4XOJjRcDcfR3dKAt3b6cTEYQSgSBxY5W+jVcAyrmryIxDWEUrOFarpMK48ugWuzMfiss4UGGtC3OpA9W+hMDOtWNkLTJa7NxtDsdSOh6VjdsrDZQlf6PJiNa0hoGlpSs4Uaf+ebLVRKmXacrLOFTkzH0Bnwoj81m9nVcAyu+WYLjavoaF7UbKFFPyE5V1vlDIO0GEW0n6LaK9tq8Vg/hVWyrQJzs4VOTMfQ1uTBTDyBphyzhbY0OhCKFJot1IO+rkDWbKFGPxPXdAgIxFQVLocDE6nZqzU5N1voikYHZuMSs3ENjW4HFAFoqdlCdSnhczsxG0/OFnptNoGVPjc8TgElNVvoTFTDqmY33A4BXQKaLqHq0pwJteBsoTMxtPrc0KWGZmO20HCqTlIzqDodybwUAcRViYmZGFZYZgttb57rX8eno2hrSs4WOhKMwuVQEIrG4fe60ehWMGvMFur3osXnwkgwObuoMTvo+HT2665AcrbQ0VAUnQFvWn0bjHo39i80W2ihbWxmS1vlbKGFcbZQW5TyRsj74VAIgTu/+nxRiT3+W7ehlj5vUs2q3RloiBaopgbXipXvYoWozGwZXCOqEFs+BBJVANsq1Qu2VaoXHFyjesHBNap7S+62UCIiIiIiIqoBihNCiKJ+uq9bW+2oiYiK5qx2AERERERERLQE6WpJ33YjIqo3/OYaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tERERERERUt7qvWwshRFE/3detrXbYRLSEOKsdABEREREREVGpLl98E3d+9fmi9nn8t24rUzREtBzxm2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQlqqnBNSHENiHEa0KI00KIP6h2PERERERERERERIU4qx2AQQjhAPAVAL8I4CKAnwohDkkpXykmHVXVcf5qCJE44LKUThGALuf+jiUARQEcGcOLup5cbt3eKqHO7ReNSzic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rpJ5EVFpnE4Fu27uRm97E8ZCMaz0uSEhsa2/k/1kjeK5leqF9fwyGoyiM+BFX1fAttlCeX1PREtFuQfXbpdSXhJCtAP4nhDiVetKKaVMDbwtmBDiHiRvG8XatWvti7TCFEVgQ1sTNrQ1VTuUmrDQ+qinelsqbZVqk9vtwGDPSlvSmq+t2pkX0WLwvJqf06ng5utWVDsMSllIW+W5lWrFfO3VOL/cfF158q+n63sionzKeluolPJS6vc4gG8DeAeAMeN2z9Tv8dTmlwBYT9lrUssy03xESjkopRxsa2srZ/hEi8K2SvWCbZXqBdsq1Qu2VaonbK9ERItXtsE1IYRPCNFs/A3g/QCGABwC8OupzX4dwFOpvw8BuDs1a+itAIKW20eJiIiIiIiIiIhqTjlvC+0A8G0hhJHP/5BSPiOE+CmAJ4QQnwJwHsDu1PbfAfBBAKcBzAL4jTLGRkREREREREREtGhlG1yTUp4BcHOO5ZMAtuZYLgF8plzxEBERERERERER2a2sz1wjIiIiIiIiIiJayji4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQl4uAaERERERERERFRiTi4RkREREREREREVCIOrhEREREREREREZWIg2tEREREREREREQlKvvgmhDCIYT4mRDi6dTr9UKInwghTgshHhdCuFPLPanXp1Pre8odGxERERERERER0WJU4ptr9wI4aXn9RQBfklLeAOAagE+lln8KwLXU8i+ltiMiIiIiIiIiIqpZZR1cE0KsAfDLAP4+9VoA2ALgm6lNHgWwK/X3ztRrpNZvTW1PRERERERERERUk8r9zbW/AvD7APTU61YAU1JKNfX6IoDu1N/dAN4EgNT6YGp7IiIiIiIiIiKimlS2wTUhxHYA41LKF21O9x4hxFEhxNGJiQk7kyayFdsq1Qu2VaoXbKtUL9hWqZ6wvRIRLV45v7n2nwDsEEKcA/ANJG8HfQhAixDCmdpmDYBLqb8vAbgOAFLrAwAmMxOVUj4ipRyUUg62tbWVMXyixWFbpXrBtkr1gm2V6gXbKtUTtlciosUr2+CalPIPpZRrpJQ9AD4G4Fkp5V0AfgDgI6nNfh3AU6m/D6VeI7X+WSmlLFd8REREREREREREi1WJ2UIzfQ7AfUKI00g+U+1rqeVfA9CaWn4fgD+oQmxEREREREREREQL5px/k8WTUv4QwA9Tf58B8I4c20QBfLQS8RAREREREREREdmhGt9cIyIiIiIiIiIiWhI4uEZERERERERERFQiDq4RERERERERERGViINrREREREREREREJeLgGhERERERERERUYk4uEZERERERERERFQiDq4RERERERERERGViINrREREREREREREJeLgGhERERERERERUYk4uEZERERERERERFQiDq4RERERERERERGViINrREREREREREREJeLgGhERERERERERUYk4uEZERERERERERFQiDq4RERERERERERGViINrREREREREREREJeLgGhERERERERERUYk4uEZERERERERERFQiDq4RERERERERERGViINrREREREREREREJSrb4JoQwiuE+D9CiGNCiGEhxP7U8vVCiJ8IIU4LIR4XQrhTyz2p16dT63vKFRsREREREREREZEdyvnNtRiALVLKmwFsBrBNCHErgC8C+JKU8gYA1wB8KrX9pwBcSy3/Umo7IiIiIiIiIiKimlW2wTWZNJN66Ur9SABbAHwztfxRALtSf+9MvUZq/VYhhChXfERERERERERERItV1meuCSEcQoiXAYwD+B6ANwBMSSnV1CYXAXSn/u4G8CYApNYHAbTmSPMeIcRRIcTRiYmJcoZPtChsq1Qv2FapXrCtUr1gW6V6wvZKRLR4ZR1ck1JqUsrNANYAeAeAt9iQ5iNSykEp5WBbW9tikyMqG7ZVqhdsq1Qv2FapXrCtUj1heyUiWryKzBYqpZwC8AMA7wLQIoRwplatAXAp9fclANcBQGp9AMBkJeIjIiIiIiIiIiIqRTlnC20TQrSk/m4A8IsATiI5yPaR1Ga/DuCp1N+HUq+RWv+slFKWKz4iIiIiIiIiIqLFcs6/Scm6ADwqhHAgOYj3hJTyaSHEKwC+IYR4AMDPAHwttf3XAPyTEOI0gKsAPlbG2IiIiIiIiIiIiBZtwYNr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\n",
      "text/plain": [
       "<Figure size 1260x1260 with 35 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(train_data,corner=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "139648bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_data['Sex'].value_counts().plot(kind='barh',)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "91872cd7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_data['Survived'].value_counts().plot(kind='barh')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "9a418137",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    549\n",
       "1    342\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8bb2c2dd",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    61.616162\n",
       "1    38.383838\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 1 servival rate , 0 death rate\n",
    "train_data['Survived'].value_counts()/len(train_data)*100"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71bd5f4f",
   "metadata": {},
   "source": [
    "# Servive rate in Man & Women\n",
    "#### Male"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c5c4fd99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12.247191011235955"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.loc[train_data.Sex=='male']['Survived'].sum()/(len(train_data)-1)*100"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a75bbd3b",
   "metadata": {},
   "source": [
    "#### Female"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "5bceee32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "26.179775280898877"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.loc[train_data.Sex=='female']['Survived'].sum()/(len(train_data)-1)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c69fbbe",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "61e043ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.barh(['male','female'],[12.247191011235955,26.179775280898877],color=['red','blue'])\n",
    "plt.xlabel('Sirvival rate')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "99349c16",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "840df9d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "16034991",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "53fe8a67",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = [\"Pclass\",\"Sex\",\"SibSp\",\"Parch\"]\n",
    "X=pd.get_dummies(train_data[features])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1e738159",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test=pd.get_dummies(test_data[features])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3e6f5b1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y=train_data['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "0be0cce6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_depth=5, random_state=1)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X,Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e1f9d770",
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_Y=model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "25b29e11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',\n",
       "       'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "4a5147e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "406"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(predicted_Y==Y_test_real_values.Survived)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "4eb9ad70",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 2 columns):\n",
      " #   Column       Non-Null Count  Dtype\n",
      "---  ------       --------------  -----\n",
      " 0   PassengerId  418 non-null    int64\n",
      " 1   Survived     418 non-null    int64\n",
      "dtypes: int64(2)\n",
      "memory usage: 6.7 KB\n"
     ]
    }
   ],
   "source": [
    "Y_test_real_values.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "672fb61a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "97.3621103117506"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(sum(predicted_Y==Y_test_real_values.Survived))/(len(Y_test_real_values)-1)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "23f01176",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived\n",
       "0            892         0\n",
       "1            893         1\n",
       "2            894         0\n",
       "3            895         0\n",
       "4            896         1\n",
       "..           ...       ...\n",
       "413         1305         0\n",
       "414         1306         1\n",
       "415         1307         0\n",
       "416         1308         0\n",
       "417         1309         0\n",
       "\n",
       "[418 rows x 2 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_test_real_values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "fa0141a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,\n",
       "       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,\n",
       "       1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,\n",
       "       1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,\n",
       "       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,\n",
       "       0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,\n",
       "       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,\n",
       "       0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,\n",
       "       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,\n",
       "       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,\n",
       "       1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predicted_Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "69deebfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_test_real_values['predicted_Y']=predicted_Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "bc47c12c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>predicted_Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  predicted_Y\n",
       "0            892         0            0\n",
       "1            893         1            1\n",
       "2            894         0            0\n",
       "3            895         0            0\n",
       "4            896         1            1\n",
       "..           ...       ...          ...\n",
       "413         1305         0            0\n",
       "414         1306         1            1\n",
       "415         1307         0            0\n",
       "416         1308         0            0\n",
       "417         1309         0            0\n",
       "\n",
       "[418 rows x 3 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_test_real_values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "31e90c75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "406"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(Y_test_real_values['Survived']==Y_test_real_values['predicted_Y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "c3786ed6",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_predicted_values=Y_test_real_values.drop(['Survived'],axis=1,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "ff450c70",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_predicted_values.to_csv('servived_predict.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "1d1be41e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>predicted_Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  predicted_Y\n",
       "0            892         0            0\n",
       "1            893         1            1\n",
       "2            894         0            0\n",
       "3            895         0            0\n",
       "4            896         1            1\n",
       "..           ...       ...          ...\n",
       "413         1305         0            0\n",
       "414         1306         1            1\n",
       "415         1307         0            0\n",
       "416         1308         0            0\n",
       "417         1309         0            0\n",
       "\n",
       "[418 rows x 3 columns]"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_test_real_values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "bca92b62",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " gender_submission.csv   test.csv                    \u001b[0m\u001b[01;31mtitanic.zip\u001b[0m\r\n",
      " servived_predict.csv   'titanic data under.ipynb'   train.csv\r\n"
     ]
    }
   ],
   "source": [
    "ls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "ac2a50f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>predicted_Y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  predicted_Y\n",
       "0            892            0\n",
       "1            893            1\n",
       "2            894            0\n",
       "3            895            0\n",
       "4            896            1\n",
       "..           ...          ...\n",
       "413         1305            0\n",
       "414         1306            1\n",
       "415         1307            0\n",
       "416         1308            0\n",
       "417         1309            0\n",
       "\n",
       "[418 rows x 2 columns]"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.read_csv('servived_predict.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4cb4c1d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
