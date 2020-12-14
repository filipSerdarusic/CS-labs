// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <sstream>
// #include <map>
// #include <chrono>
#include <bits/stdc++.h>

using namespace std;

int main() {	

	int brKosara;
	cin >> brKosara;

	float s;
	cin >> s;
	
	int prag = s * brKosara;

	int brPretinaca;
	cin >> brPretinaca;

	// brojac predmeta
	map <int,int> brPredmeta;

	string line;
	string item;

	vector <vector<int>> data;

	// prvi prolaz
	while(getline(cin, line)) {
		
		vector <int> numbers;
		stringstream ss(line);

		while(getline(ss, item, ' ')) {

			numbers.push_back(stoi(item));
			brPredmeta[stoi(item)]++;
		}
		
		data.push_back(numbers);
	}
	
	// pretinci za funkciju sazimanja - polje velicine brPretinaca
	vector <int> pretinci(brPretinaca, 0);

	// drugi prolaz - sazimanje
	for (auto kosara : data) {

		// broj predmeta u kosari
		int n = kosara.size(); 

		// za svaki par predmeta unutar kosare
		for(int i=0; i<n; i++) {
			for(int j=i+1; j<n; j++) {

				// sazmi par predmeta u pretinac
				// oba predmeta moraju biti cesta
				if ((brPredmeta[kosara[i]] >= prag) && 
					(brPredmeta[kosara[j]] >= prag)) {

					int k = ((kosara[i] * brPredmeta.size()) + kosara[j]) % brPretinaca;

					pretinci[k]++;
				}
			}
		}
	}

	// mapa - kljuc par predmeta [i, j], vrijednost broj ponavljanja
	map <pair<int,int>, int> parovi;
	
	// treci prolaz - brojanje parova
	for (auto kosara : data) {
		
		int n = kosara.size();
		for (int i=0; i<n; i++) {

			for (int j=i+1; j<n; j++) {

				//oba predmeta moraju biti cesta
				if ((brPredmeta[kosara[i]] >= prag) && 
					(brPredmeta[kosara[j]] >= prag)) {

					int k = ((kosara[i] * brPredmeta.size()) + kosara[j]) % brPretinaca;

					// parovi moraju biti u cestom pretincu
					if (pretinci[k] >= prag)
						parovi[make_pair (kosara[i],kosara[j])]++;
				}
			}
		}
	}

	// Broj cestih predmeta
	int brCestih = 0;
	for (auto num : brPredmeta) {
		if (num.second >= prag)
			brCestih++;
	}
	
	int A = brCestih*(brCestih-1)/2;

	cout << A << endl;
	cout << parovi.size() << endl;
	
	// Silazno sortirani brojevi ponavljanja cestih parova
	vector <int> values;
	for (auto item : parovi) {
		if (item.second >= prag)
			values.push_back(item.second);
	}
	sort(values.begin(), values.end(), greater<int>());

	for (auto item : values)
		cout << item << "\n";

}