import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../menudetail/menudetail_screens.dart';

class MenuSearchDelegate extends SearchDelegate {
  String? selectedCategory; // ✅ ประกาศตัวแปรให้แน่ใจว่าใช้งานได้
  String? selectedBestSeller;
  RangeValues priceRange = const RangeValues(0, 500); // ✅ ตัวกรองราคา

  @override
  List<Widget> buildActions(BuildContext context) {
    return [
      IconButton(
        icon: Icon(Icons.filter_list),
        onPressed: () {
          _showFilterDialog(context);
        },
      ),
    ];
  }

  @override
  Widget buildLeading(BuildContext context) {
    return IconButton(
      icon: Icon(Icons.arrow_back),
      onPressed: () {
        close(context, null);
      },
    );
  }

  @override
  Widget buildResults(BuildContext context) {
    return _buildSearchResults();
  }

  @override
  Widget buildSuggestions(BuildContext context) {
    return _buildSearchResults();
  }

  Widget _buildSearchResults() {
    return StreamBuilder<QuerySnapshot>(
      stream: FirebaseFirestore.instance.collection('Foods').snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
          return const Center(child: Text("ไม่มีเมนูอาหารที่ตรงกับคำค้นหา"));
        }

        final results = snapshot.data!.docs.where((doc) {
          final data = doc.data() as Map<String, dynamic>;

          final name = (data['name'] ?? '').toString().toLowerCase();
          final category = (data['category'] ?? '').toString();
          final isBestSeller = data['isBestSeller'] ?? false;
          final isAvailable = data['isAvailable'] ?? true;

          final price = (data['price'] is num)
              ? (data['price'] as num).toDouble()
              : double.tryParse(data['price'].toString()) ?? 0.0;

          bool matchesPrice = price >= priceRange.start && price <= priceRange.end;
          bool matchesQuery = query.isNotEmpty ? name.contains(query.toLowerCase()) : true;
          bool matchesCategory = selectedCategory == null || category == selectedCategory;
          bool matchesBestSeller = selectedBestSeller == null ||
              (selectedBestSeller == "เฉพาะเมนูขายดี" && isBestSeller) ||
              (selectedBestSeller == "ทั้งหมด");

          return matchesQuery && matchesPrice && matchesCategory && matchesBestSeller && isAvailable;
        }).toList();

        return ListView.builder(
          itemCount: results.length,
          itemBuilder: (context, index) {
            final data = results[index].data() as Map<String, dynamic>;
            final isAvailable = data['isAvailable'] ?? true;

            return Container(
              color: index.isEven ? Colors.grey[200] : Colors.white,
              child: ListTile(
                title: Text(
                  data['name'],
                  style: TextStyle(
                    color: isAvailable ? Colors.black : Colors.grey,
                  ),
                ),
                subtitle: Text(
                  "ราคา: ${data['price']} บาท",
                  style: TextStyle(
                    color: isAvailable ? Colors.black : Colors.grey,
                  ),
                ),
                trailing: data['isBestSeller'] == true
                    ? Icon(Icons.star, color: Colors.orange)
                    : null,
                enabled: isAvailable,
                onTap: isAvailable
                    ? () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => MenuDetailScreen(
                              name: data['name'],
                              image: data['imageBase64'] ?? '',
                              price: data['price'].toString(),
                              category: data['category'] ?? 'ไม่ระบุหมวดหมู่',
                            ),
                          ),
                        );
                      }
                    : null,
              ),
            );
          },
        );
      },
    );
  }

  // ✅ แสดงตัวกรอง
  void _showFilterDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          title: const Text("ตัวกรอง", style: TextStyle(fontWeight: FontWeight.bold)),
          content: StatefulBuilder(
            builder: (context, setState) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // ✅ Dropdown เลือกประเภทอาหาร
                  DropdownButtonFormField<String>(
                    value: selectedCategory,
                    decoration: const InputDecoration(labelText: "ประเภท"),
                    items: [
                      'ผัด', 'ต้ม', 'ทอด', 'ยำ / ลาบ', 'แกง', 'ส้มตำ', 'ก๋วยเตี๋ยว', 'เครื่องดื่ม'
                    ].map((type) => DropdownMenuItem(value: type, child: Text(type))).toList(),
                    onChanged: (value) {
                      setState(() => selectedCategory = (value == "ทั้งหมด") ? null : value);
                    },
                  ),

                  // ✅ ตัวกรองราคา
                  const SizedBox(height: 10),
                  const Text("ช่วงราคา (บาท)", style: TextStyle(fontWeight: FontWeight.bold)),
                  RangeSlider(
                    values: priceRange,
                    min: 0,
                    max: 500,
                    divisions: 10,
                    labels: RangeLabels(
                      priceRange.start.toStringAsFixed(0),
                      priceRange.end.toStringAsFixed(0),
                    ),
                    onChanged: (newRange) {
                      setState(() => priceRange = newRange);
                    },
                  ),

                  // ✅ ตัวกรองเมนูขายดี
                  DropdownButtonFormField<String>(
                    value: selectedBestSeller,
                    decoration: const InputDecoration(labelText: "เมนูขายดี"),
                    items: ["ทั้งหมด", "เฉพาะเมนูขายดี"]
                        .map((option) => DropdownMenuItem(value: option, child: Text(option)))
                        .toList(),
                    onChanged: (value) {
                      setState(() => selectedBestSeller = (value == "ทั้งหมด") ? null : value);
                    },
                  ),
                ],
              );
            },
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
                showResults(context);
              },
              child: const Text("ตกลง", style: TextStyle(fontSize: 16)),
            ),
          ],
        );
      },
    );
  }
}
