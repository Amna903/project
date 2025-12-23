{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "TN-smBQiyiEB"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "T_qhXfZz4Pzp"
   },
   "source": [
    "## Demand Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "ed8yd2gfvowt"
   },
   "outputs": [],
   "source": [
    "DemandDF = pd.read_csv('WoolworthsDemand2025.csv', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "TuT-T6PAys3l"
   },
   "outputs": [],
   "source": [
    "week_numbers = (np.arange(28) % 7) + 1\n",
    "\n",
    "SaturdayDF = DemandDF.T.iloc[week_numbers==6].T\n",
    "WeekdaysDF = DemandDF.T.iloc[week_numbers<6].T\n",
    "\n",
    "WeekStoreMean = WeekdaysDF.mean(axis=1)\n",
    "SaturdayMean = SaturdayDF.mean(axis=1)\n",
    "\n",
    "WeekStoreSD = WeekdaysDF.std(axis=1)\n",
    "SaturdaySD = SaturdayDF.std(axis=1)\n",
    "\n",
    "WeekStore75 = WeekStoreMean + 0.675*WeekStoreSD\n",
    "Saturday75 = SaturdayMean + 0.675*SaturdaySD\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 677
    },
    "id": "7794f3de",
    "outputId": "df73bf0f-57b0-44f8-b11e-4b304084aa49"
   },
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
       "      <th>Weekday</th>\n",
       "      <th>Saturday</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Store</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>FreshChoice Cannons Creek</th>\n",
       "      <td>2.587546</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>FreshChoice Cuba Street</th>\n",
       "      <td>2.246321</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>FreshChoice Woburn</th>\n",
       "      <td>2.194569</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Metro Cable Car Lane</th>\n",
       "      <td>2.587546</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Aotea</th>\n",
       "      <td>3.034719</td>\n",
       "      <td>1.175000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Crofton Downs</th>\n",
       "      <td>3.675902</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Johnsonville</th>\n",
       "      <td>3.852079</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Johnsonville Mall</th>\n",
       "      <td>3.035806</td>\n",
       "      <td>1.396263</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Karori</th>\n",
       "      <td>2.908907</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Kilbirnie</th>\n",
       "      <td>3.099753</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Lower Hutt</th>\n",
       "      <td>3.106250</td>\n",
       "      <td>1.779423</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Maidstone</th>\n",
       "      <td>3.860129</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Newtown</th>\n",
       "      <td>3.535806</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Petone</th>\n",
       "      <td>3.087546</td>\n",
       "      <td>0.889711</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Porirua</th>\n",
       "      <td>3.622567</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Queensgate</th>\n",
       "      <td>2.803738</td>\n",
       "      <td>0.587500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Tawa</th>\n",
       "      <td>2.548753</td>\n",
       "      <td>0.889711</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Upper Hutt</th>\n",
       "      <td>2.823028</td>\n",
       "      <td>1.551135</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Woolworths Wainuiomata</th>\n",
       "      <td>3.522340</td>\n",
       "      <td>1.087500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                               Weekday  Saturday\n",
       "Store                                           \n",
       "FreshChoice Cannons Creek     2.587546  0.000000\n",
       "FreshChoice Cuba Street       2.246321  0.000000\n",
       "FreshChoice Woburn            2.194569  0.000000\n",
       "Metro Cable Car Lane          2.587546  0.000000\n",
       "Woolworths Aotea              3.034719  1.175000\n",
       "Woolworths Crofton Downs      3.675902  0.587500\n",
       "Woolworths Johnsonville       3.852079  0.587500\n",
       "Woolworths Johnsonville Mall  3.035806  1.396263\n",
       "Woolworths Karori             2.908907  0.587500\n",
       "Woolworths Kilbirnie          3.099753  0.587500\n",
       "Woolworths Lower Hutt         3.106250  1.779423\n",
       "Woolworths Maidstone          3.860129  0.000000\n",
       "Woolworths Newtown            3.535806  0.587500\n",
       "Woolworths Petone             3.087546  0.889711\n",
       "Woolworths Porirua            3.622567  0.587500\n",
       "Woolworths Queensgate         2.803738  0.587500\n",
       "Woolworths Tawa               2.548753  0.889711\n",
       "Woolworths Upper Hutt         2.823028  1.551135\n",
       "Woolworths Wainuiomata        3.522340  1.087500"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "demand75 = pd.DataFrame({\n",
    "    'Weekday': WeekStore75,\n",
    "    'Saturday': Saturday75,\n",
    "})\n",
    "\n",
    "display(demand75)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "VsYBFvNr4YBf"
   },
   "source": [
    "## Routes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "Rh-zESag4azk"
   },
   "outputs": [],
   "source": [
    "TravelDF = pd.read_csv('WoolworthsDurations2025.csv', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Weekday 8:00 shift - Mean mult: 1.60x, CV: 0.20, Lognorm(s=0.198, scale=1.569)\n",
      "Weekday 13:00 shift - Mean mult: 1.38x, CV: 0.15, Lognorm(s=0.149, scale=1.365)\n",
      "Saturday 8:00 shift - Mean mult: 1.09x, CV: 0.10, Lognorm(s=0.100, scale=1.085)\n",
      "Saturday 13:00 shift - Mean mult: 1.53x, CV: 0.15, Lognorm(s=0.149, scale=1.513)\n",
      "Example: Base 9.7 min -> Sim 16.7 min (mult=1.73x)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from scipy.stats import lognorm\n",
    "\n",
    "# Scenarios: (day_type, shift_hour) -> (mean_multiplier, cv)\n",
    "TRAFFIC_PARAMS = {\n",
    "    ('weekday', 8):  (1.60, 0.20),   # Morning peak\n",
    "    ('weekday', 13): (1.38, 0.15),   # Midday\n",
    "    ('saturday', 8): (1.09, 0.10),   # Morning low\n",
    "    ('saturday', 13): (1.53, 0.15)   # Midday moderate\n",
    "}\n",
    "\n",
    "def get_lognorm_params(mean_mult, cv):\n",
    "    sigma = np.sqrt(np.log(1 + cv**2))\n",
    "    mu = np.log(mean_mult) - 0.5 * sigma**2\n",
    "    return sigma, np.exp(mu)\n",
    "\n",
    "LOGNORM_PARAMS = {}\n",
    "for (day, hour), (mean_mult, cv) in TRAFFIC_PARAMS.items():\n",
    "    sigma, scale = get_lognorm_params(mean_mult, cv)\n",
    "    LOGNORM_PARAMS[(day, hour)] = (sigma, scale)\n",
    "    print(f\"{day.capitalize()} {hour}:00 shift - Mean mult: {mean_mult:.2f}x, CV: {cv:.2f}, \"\n",
    "          f\"Lognorm(s={sigma:.3f}, scale={scale:.3f})\")\n",
    "\n",
    "def sample_route_multipliers(day_type, shift_hour, num_legs, correlated=False):\n",
    "    sigma, scale = LOGNORM_PARAMS[(day_type, shift_hour)]\n",
    "    if correlated:\n",
    "        return np.full(num_legs, lognorm.rvs(s=sigma, scale=scale, size=1))\n",
    "    return lognorm.rvs(s=sigma, scale=scale, size=num_legs)\n",
    "\n",
    "# Example\n",
    "np.random.seed(42)  # For reproducibility\n",
    "example_base = TravelDF.loc['CentrePort Wellington', 'Woolworths Newtown']\n",
    "mult = sample_route_multipliers('weekday', 8, 1)[0]\n",
    "sim_time = example_base * mult\n",
    "print(f\"Example: Base {example_base / 60:.1f} min -> Sim {sim_time / 60:.1f} min (mult={mult:.2f}x)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "s0uYFIurO-zu"
   },
   "source": [
    "#### Initialise Route Class"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "n9tD0_5CQrMZ"
   },
   "source": [
    "#### Cheapest Insertion Routes:\n",
    "Insertion until total demand surpasses 9, or total time exceeds 10800"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "tf0nLmA4RFVi"
   },
   "outputs": [],
   "source": [
    "def cheapest_insertion_route(travelDF, demandDF, remaining_stores = None,is_saturday=False, max_boxes=9, max_time=10800):\n",
    "    route = Route(travelDF, demandDF, saturday=is_saturday)\n",
    "    route.route = [\"CentrePort Wellington\", \"CentrePort Wellington\"]\n",
    "    route.travelTime = 0\n",
    "    route.demand = 0\n",
    "    if remaining_stores is None:\n",
    "        remaining_stores = set(demandDF.index)\n",
    "\n",
    "    stores_added_in_iteration = True\n",
    "    while stores_added_in_iteration:\n",
    "        stores_added_in_iteration = False\n",
    "        best_store = None\n",
    "        best_position = None\n",
    "        best_time_increase = float('inf')\n",
    "\n",
    "        for store in remaining_stores:\n",
    "            store_demand = demandDF.loc[store]['Saturday' if is_saturday else 'Weekday']\n",
    "            if route.demand + store_demand > max_boxes:\n",
    "                continue\n",
    "\n",
    "            for pos in range(1, len(route.route)):\n",
    "                prev_stop = route.route[pos - 1]\n",
    "                next_stop = route.route[pos]\n",
    "\n",
    "                added_time = (\n",
    "                    travelDF.loc[prev_stop, store] +\n",
    "                    travelDF.loc[store, next_stop] -\n",
    "                    travelDF.loc[prev_stop, next_stop]\n",
    "                )\n",
    "\n",
    "                if (route.travelTime + added_time) > max_time:\n",
    "                    continue\n",
    "\n",
    "                if added_time < best_time_increase:\n",
    "                    best_store = store\n",
    "                    best_position = pos\n",
    "                    best_time_increase = added_time\n",
    "\n",
    "        if best_store is not None:\n",
    "            route.addStop(best_store, best_position)\n",
    "            remaining_stores.remove(best_store)\n",
    "            stores_added_in_iteration = True\n",
    "\n",
    "    if len(route.route) > 2: # Only return route if stores were added\n",
    "        return route\n",
    "    else:\n",
    "        return False # Return False if no stores could be added to this route"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "v0ncNfMJdRRa"
   },
   "source": [
    "### Route Generation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "4yaDXRz6MLoD"
   },
   "outputs": [],
   "source": [
    "def generate_all_feasible_routes(travelDF, demandDF, saturday=False,\n",
    "                                 max_boxes=9, max_time=10800,\n",
    "                                 current_route=[\"CentrePort Wellington\"],\n",
    "                                 all_routes=None, min_return_time=None, demand_dict=None):\n",
    "    if all_routes is None:\n",
    "        all_routes = []\n",
    "\n",
    "    if min_return_time is None:\n",
    "        depot = \"CentrePort Wellington\"\n",
    "        min_return_time = {store: travelDF.loc[store, depot] for store in demandDF.index}\n",
    "    if demand_dict is None:\n",
    "        demand_dict = {store: (row.iloc[0], row.iloc[1]) for store, row in demandDF.iterrows()}\n",
    "\n",
    "    end_node = current_route[-1]\n",
    "\n",
    "    completed_route = Route(travelDF, demandDF, saturday=saturday)\n",
    "    completed_route.addList(current_route[1:])\n",
    "\n",
    "\n",
    "    if end_node != \"CentrePort Wellington\":\n",
    "        completed_route.complete_round_trip()\n",
    "        completed_route.genCost()\n",
    "        if completed_route.travelTime <= max_time:\n",
    "            if completed_route not in all_routes:\n",
    "\n",
    "                all_routes.append(completed_route)\n",
    "            else:\n",
    "                for i, r in enumerate(all_routes):\n",
    "                    if r == completed_route:\n",
    "                        if completed_route.travelTime < r.travelTime:\n",
    "                            all_routes[i] = completed_route\n",
    "                        break\n",
    "\n",
    "    new_route = Route(travelDF, demandDF, saturday=saturday)\n",
    "    new_route.addList(current_route[1:])\n",
    "\n",
    "    for store in demandDF.index:\n",
    "        if store not in new_route.route:\n",
    "            if saturday and demand_dict[store][1] == 0:\n",
    "                continue\n",
    "\n",
    "            new_demand = new_route.testDemand(store)\n",
    "            if new_demand <= max_boxes:\n",
    "                new_time = new_route.testTime(store)\n",
    "\n",
    "                if new_time + min(min_return_time.values()) > max_time:\n",
    "                    continue\n",
    "\n",
    "                if new_time <= max_time:\n",
    "                    generate_all_feasible_routes(\n",
    "                        travelDF, demandDF, saturday, max_boxes, max_time,\n",
    "                        current_route + [store], all_routes,\n",
    "                        min_return_time=min_return_time,\n",
    "                        demand_dict=demand_dict\n",
    "                    )\n",
    "\n",
    "    return all_routes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "5ysFYpxFVyni",
    "outputId": "c56cb2b1-566d-4003-d09e-429494954fba"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "found 598 routes\n"
     ]
    }
   ],
   "source": [
    "all_weekday_routes = generate_all_feasible_routes(TravelDF, demand75, max_time=(10*3600))\n",
    "print(f\"found {len(all_weekday_routes)} routes\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "GLizxE4GMSQQ",
    "outputId": "2178954f-1c2d-44d0-de49-2947bc2b5c11"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The route with the maximum demand on a weekday is: 8.997535082965873\n",
      "The route with the maximum time on a weekday is: 15393.753337568616\n"
     ]
    }
   ],
   "source": [
    "maxDemand = max(all_weekday_routes, key=lambda route: route.demand)\n",
    "print(f\"The route with the maximum demand on a weekday is: {maxDemand.demand}\")\n",
    "maxTime = max(all_weekday_routes, key=lambda route: route.travelTime)\n",
    "print(f\"The route with the maximum time on a weekday is: {maxTime.travelTime}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "YW_JYTRtV0J1",
    "outputId": "02b26a0d-6488-4a7d-b801-50ecc89a0ce5"
   },
   "outputs": [],
   "source": [
    "all_saturday_routes = generate_all_feasible_routes(TravelDF, demand75, True)\n",
    "print(f\"found {len(all_saturday_routes)} routes\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "mt4PabM2ufJM",
    "outputId": "0a6edf26-dffe-4b8f-9945-22614fbf1382"
   },
   "outputs": [],
   "source": [
    "maxsatDemand = max(all_saturday_routes, key=lambda route: route.demand)\n",
    "print(f\"The route with the maximum demand on a saturday is: {maxsatDemand.demand}\")\n",
    "maxsatTime = max(all_saturday_routes, key=lambda route: route.travelTime)\n",
    "print(f\"The route with the maximum time on a saturday is: {maxsatTime.travelTime}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "01d16864",
    "outputId": "2a593c46-a980-4c47-8215-8363fee8ad75"
   },
   "outputs": [],
   "source": [
    "print(f\"found {len(all_weekday_routes)} feasible routes on the weekday, and {len(all_saturday_routes)} on the saturday.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 677
    },
    "id": "DIS5P1PGo6DY",
    "outputId": "68fe4695-f6ff-42d7-b0f7-e16dd059a205"
   },
   "outputs": [],
   "source": [
    "demand75"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "l2Fgry4S4LYb"
   },
   "source": [
    "## Linear Programming"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "OjUbfWqbehDM",
    "outputId": "f3c5f0d2-3576-45e8-952e-97fdedfe543c"
   },
   "outputs": [],
   "source": [
    "%pip install PuLP"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-tmqwS9aPhFG"
   },
   "source": [
    "### Linear Program"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "FcxDyDparhcr"
   },
   "outputs": [],
   "source": [
    "from pulp import *\n",
    "import gurobipy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "cw4zdNhxW6Ss"
   },
   "outputs": [],
   "source": [
    "prob = LpProblem(\"Total_Cost_For_Woolworths\", LpMinimize)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "1D9SIprCW9AB"
   },
   "outputs": [],
   "source": [
    "# Generate Weekday Routes\n",
    "weekday_routes = {f\"W_Route{m}\": all_weekday_routes[m] for m in range(len(all_weekday_routes))}\n",
    "\n",
    "# Generate Saturday Routes\n",
    "saturday_routes = {f\"S_Route{m}\": all_saturday_routes[m] for m in range(len(all_saturday_routes))}\n",
    "\n",
    "# Combine all routes\n",
    "all_routes = {**weekday_routes, **saturday_routes}\n",
    "all_routes_keys = list(all_routes.keys())\n",
    "\n",
    "# Decision variables\n",
    "Selected = LpVariable.dicts(\"Selected\", all_routes_keys, 0, 1, LpBinary)\n",
    "Van = LpVariable.dicts(\"Van\", all_routes_keys, 0, 1, LpBinary)\n",
    "Overtime = LpVariable.dicts(\"Overtime\", all_routes_keys, 0, None, LpContinuous)\n",
    "SubCost = LpVariable.dicts(\"SubCost\", all_routes_keys, 0, None, LpInteger)\n",
    "Z = LpVariable.dicts(\"Z\", all_routes_keys, 0, 1, LpBinary)  # Selected * Van\n",
    "W = LpVariable.dicts(\"W\", all_routes_keys, 0, None, LpContinuous)  # Selected * SubCost\n",
    "Y = LpVariable.dicts(\"Y\", all_routes_keys, 0, 1, LpBinary)\n",
    "Fleet = LpVariable(\"Fleet\", 0, 4, LpInteger)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "-MwbIF6EajrV"
   },
   "outputs": [],
   "source": [
    "# Create route coverage dictionary\n",
    "stores_to_cover = demand75.index.tolist()\n",
    "\n",
    "# Combined\n",
    "AllRouteDict = {}\n",
    "\n",
    "for store in stores_to_cover:\n",
    "    AllRouteDict[store] = {}\n",
    "    for route_key in all_routes_keys:\n",
    "        actual_stores_in_route = all_routes[route_key].route\n",
    "        if store in actual_stores_in_route:\n",
    "            AllRouteDict[store][route_key] = 1\n",
    "        else:\n",
    "            AllRouteDict[store][route_key] = 0\n",
    "\n",
    "# Saturday only coverage\n",
    "sat_stores_to_cover = [\n",
    "    store for store in stores_to_cover\n",
    "    if demand75.loc[store, 'Saturday'] != 0\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "i-89myQ9bN0U"
   },
   "outputs": [],
   "source": [
    "# Route data\n",
    "AllTime = pd.Series({key: route.travelTime for key, route in all_routes.items()})\n",
    "AllDemandWeekday = pd.Series({key: route.demand for key, route in weekday_routes.items()})\n",
    "AllDemandSaturday = pd.Series({key: route.demand for key, route in saturday_routes.items()})\n",
    "\n",
    "\n",
    "AllRouteData = pd.DataFrame({\n",
    "    'TravelTime': AllTime,\n",
    "    'DemandWeekday': AllDemandWeekday,\n",
    "    'DemandSaturday': AllDemandSaturday,\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "VueY2-cZgg-1"
   },
   "outputs": [],
   "source": [
    "AllRouteData = pd.DataFrame({\n",
    "    'TravelTime': {key: route.travelTime for key, route in all_routes.items()},\n",
    "    'Demand': {key: route.demand for key, route in all_routes.items()},\n",
    "    'vanCost': {key: route.vanCost for key, route in all_routes.items()},\n",
    "    'subCost': {key: route.subCost for key, route in all_routes.items()},\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "4d891Hth2Jkk",
    "outputId": "563f702a-4173-4d0e-d551-b22ef312fb85"
   },
   "outputs": [],
   "source": [
    "prob = LpProblem(\"Total_Cost_For_Woolworths\", LpMinimize)\n",
    "\n",
    "prob += (\n",
    "    lpSum(\n",
    "        Z[i] * AllRouteData.loc[i, 'vanCost']\n",
    "        + (Selected[i] - Z[i]) * AllRouteData.loc[i, 'subCost']\n",
    "        for i in all_routes_keys\n",
    "    )\n",
    "    + 100000 / ((6 / 7) * 365) * Fleet\n",
    ")\n",
    "\n",
    "\n",
    "# Linearize z[i] = Selected[i] & Van[i]\n",
    "for i in all_routes_keys:\n",
    "    prob += Z[i] <= Selected[i]\n",
    "    prob += Z[i] <= Van[i]\n",
    "    prob += Z[i] >= Selected[i] + Van[i] - 1\n",
    "\n",
    "# Demand constraint\n",
    "BigDemand = 15\n",
    "for i in all_routes.keys():\n",
    "    prob += AllRouteData.loc[i,'Demand'] - 5*Van[i] <= 4 + BigDemand*(1 - Selected[i])\n",
    "\n",
    "\n",
    "# Van / Fleet constraint\n",
    "# This constraint needs to be split for weekday and saturday selections\n",
    "prob += lpSum(Z[i] for i in weekday_routes.keys()) <= 2 * Fleet\n",
    "prob += lpSum(Z[i] for i in saturday_routes.keys()) <= 2 * Fleet\n",
    "\n",
    "\n",
    "# Partitioning constraint (link routes to selected)\n",
    "# This constraint needs to be split for weekday and saturday selections\n",
    "for store in stores_to_cover:\n",
    "    prob += lpSum(AllRouteDict[store][j] * Selected[j] for j in weekday_routes.keys()) == 1\n",
    "\n",
    "for store in sat_stores_to_cover:\n",
    "     prob += lpSum(AllRouteDict[store][j] * Selected[j] for j in saturday_routes.keys()) == 1\n",
    "\n",
    "\n",
    "# Write LP file and solve\n",
    "prob.writeLP(\"Woolworthsweekdayproblem.lp\")\n",
    "\n",
    "#solver = GUROBI_CMD(msg=True, timeLimit=1800)  # timeLimit optional\n",
    "solver = HiGHS_CMD(msg=True, timeLimit=1800, threads=8)\n",
    "#prob.solve(solver)\n",
    "prob.solve()\n",
    "\n",
    "finalRoutes = []\n",
    "finalWeek = []\n",
    "finalSat = []\n",
    "\n",
    "# Status\n",
    "print(\"Status:\", LpStatus[prob.status])\n",
    "\n",
    "print(f\"Fleet: {Fleet.varValue}\")\n",
    "\n",
    "# Print selected routes with stores, van/subcontractor info, and subcontractor cost\n",
    "print(\"Selected Weekday routes and details:\")\n",
    "for i in weekday_routes.keys():\n",
    "    if Selected[i].varValue >= 1:\n",
    "        stores_in_route = all_routes[i].route\n",
    "        print(f\"{i} -> Stores: {stores_in_route}\")\n",
    "        finalRoutes.append(weekday_routes[i])\n",
    "        finalWeek.append(weekday_routes[i])\n",
    "\n",
    "print(\"Selected Saturday routes and details:\")\n",
    "for i in saturday_routes.keys():\n",
    "    if Selected[i].varValue >= 1:\n",
    "        stores_in_route = all_routes[i].route\n",
    "        print(f\"{i} -> Stores: {stores_in_route}\")\n",
    "        finalRoutes.append(saturday_routes[i])\n",
    "        finalSat.append(saturday_routes[i])\n",
    "\n",
    "# Optimised objective\n",
    "print(\"Total Cost = \", value(prob.objective))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "mIAoi-yjgg-2",
    "outputId": "f929141b-56fb-4fa2-ce2f-eaeaabd4a2cb"
   },
   "outputs": [],
   "source": [
    "finalRoutes[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "zX0e8z8Ugg-2"
   },
   "outputs": [],
   "source": [
    "import folium\n",
    "import openrouteservice as ors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "4zG6Jv4Kgg-2"
   },
   "outputs": [],
   "source": [
    "ORSkey = \"REMOVED\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "dSYpITghgg-2"
   },
   "outputs": [],
   "source": [
    "locations = pd.read_csv(\"WoolworthsLocations.csv\")\n",
    "\n",
    "store_names = locations['Store'].tolist()\n",
    "coords = locations[['Long', 'Lat']] # Mapping packages work with Long, Lat arrays\n",
    "coords = coords.to_numpy().tolist() # Make the arrays into a list of lists.\n",
    "\n",
    "\n",
    "\n",
    "# Folium, however, requires Lat, Long arrays - so a reversal is needed.\n",
    "m = folium.Map(location = list(reversed(coords[2])), zoom_start=10)\n",
    "\n",
    "# NOT RUN, to plot first store.\n",
    "# folium.Marker(list(reversed(coords[0])), popup = locations.Store[0], icon = folium.Icon(color = 'black')).add_to(m)\n",
    "\n",
    "for i in range(0, len(coords)):\n",
    "    if locations.Type[i] == \"Woolworths\":\n",
    "        iconCol = \"green\"\n",
    "    elif locations.Type[i] == \"CentrePort\":\n",
    "        iconCol = \"black\"\n",
    "    elif locations.Type[i] == \"Metro\":\n",
    "        iconCol = \"orange\"\n",
    "    elif locations.Type[i] == \"FreshChoice\":\n",
    "        iconCol = \"blue\"\n",
    "    folium.Marker(list(reversed(coords[i])), popup = locations.Store[i], icon = folium.Icon(color = iconCol)).add_to(m)\n",
    "\n",
    "store_names\n",
    "store_index = {name: i for i, name in enumerate(store_names)}\n",
    "\n",
    "client = ors.Client(key=ORSkey)\n",
    "\n",
    "weekMapRoutes = []\n",
    "for route in finalWeek:\n",
    "    routeCoords = []\n",
    "    for store in route.route:\n",
    "        routeCoords.append(coords[store_index[store]])\n",
    "\n",
    "    weekMapRoutes.append(client.directions(\n",
    "        coordinates = routeCoords, # Distribution Centre to Woolworths Newtown\n",
    "        # We can have more than two coords - it will generate a path between those coords in order.\n",
    "        profile = 'driving-hgv', # can be driving-car, driving-hgv, etc.\n",
    "        format='geojson',\n",
    "        validate = False\n",
    "        ))\n",
    "\n",
    "satMapRoutes = []\n",
    "for route in finalSat:\n",
    "    routeCoords = []\n",
    "    for store in route.route:\n",
    "        routeCoords.append(coords[store_index[store]])\n",
    "\n",
    "    satMapRoutes.append(client.directions(\n",
    "        coordinates=routeCoords,\n",
    "        profile='driving-hgv',\n",
    "        format='geojson',\n",
    "        validate=False\n",
    "    ))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "TPIru31Sgg-3",
    "outputId": "29b20e06-b3a7-450e-f087-4063937f74f6"
   },
   "outputs": [],
   "source": [
    "for route in weekMapRoutes:\n",
    "    for feature in route['features']:\n",
    "        folium.PolyLine(\n",
    "            locations=[list(reversed(coord)) for coord in feature['geometry']['coordinates']],\n",
    "            color='red',\n",
    "            weight = 5,\n",
    "            opacity = 0.8\n",
    "        ).add_to(m)\n",
    "\n",
    "m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "TUuzKwuWgg-3",
    "outputId": "17dbddbb-137c-466d-9b57-1ff04d097d14"
   },
   "outputs": [],
   "source": [
    "m = folium.Map(location = list(reversed(coords[2])), zoom_start=10)\n",
    "\n",
    "for i in range(0, len(coords)):\n",
    "    if locations.Type[i] == \"Woolworths\":\n",
    "        iconCol = \"green\"\n",
    "    elif locations.Type[i] == \"CentrePort\":\n",
    "        iconCol = \"black\"\n",
    "    else:\n",
    "        continue\n",
    "    folium.Marker(list(reversed(coords[i])), popup = locations.Store[i], icon = folium.Icon(color = iconCol)).add_to(m)\n",
    "\n",
    "for route in satMapRoutes:\n",
    "    for feature in route['features']:\n",
    "        folium.PolyLine(\n",
    "            locations=[list(reversed(coord)) for coord in feature['geometry']['coordinates']],\n",
    "            color='green',\n",
    "            weight = 5,\n",
    "            opacity = 0.8\n",
    "        ).add_to(m)\n",
    "\n",
    "m"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
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
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
