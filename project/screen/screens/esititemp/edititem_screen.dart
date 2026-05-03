import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class EditItemScreen extends StatefulWidget {
  final String menuId;
  final Map<String, dynamic> menuData; // ✅ ต้องมีค่าถูกส่งมาจากหน้าอื่น

  const EditItemScreen({
    Key? key,
    required this.menuId,
    required this.menuData,
  }) : super(key: key);

  @override
  _EditItemScreenState createState() => _EditItemScreenState();
}

class _EditItemScreenState extends State<EditItemScreen> {
  TextEditingController nameController = TextEditingController();
  TextEditingController priceController = TextEditingController();
  String? selectedCategory;
  String? selectedDetails;

  @override
  void initState() {
    super.initState();
    // เติมข้อมูลเริ่มต้นในฟิลด์
    nameController.text = widget.menuData['name'] ?? '';
    priceController.text = widget.menuData['price']?.toString() ?? '';
    selectedCategory = widget.menuData['category'];
    selectedDetails = widget.menuData['details'];
  }

  Future<void> updateItem() async {
    try {
      final String name = nameController.text.trim();
      final String price = priceController.text.trim();

      if (name.isEmpty ||
          price.isEmpty ||
          selectedCategory == null ||
          selectedDetails == null) {
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text("ข้อมูลไม่ครบถ้วน"),
            content: const Text("กรุณากรอกข้อมูลให้ครบถ้วน"),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text("ปิด"),
              ),
            ],
          ),
        );
        return;
      }

      // อัปเดตข้อมูลใน Firestore
      await FirebaseFirestore.instance
          .collection('Foods')
          .doc(widget.menuId)
          .update({
        'name': name,
        'category': selectedCategory,
        'price': price,
        'details': selectedDetails,
      });

      // แสดงข้อความสำเร็จ
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("อัปเดตสำเร็จ"),
          content: const Text("ข้อมูลเมนูได้รับการอัปเดตแล้ว"),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                Navigator.pop(context); // กลับไปหน้าหลัก
              },
              child: const Text("ตกลง"),
            ),
          ],
        ),
      );
    } catch (e) {
      // แสดงข้อความข้อผิดพลาด
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("เกิดข้อผิดพลาด"),
          content: Text("ไม่สามารถอัปเดตข้อมูลได้: $e"),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("ปิด"),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: ClipRRect(
          borderRadius: const BorderRadius.only(
            bottomLeft: Radius.circular(20),
            bottomRight: Radius.circular(20),
          ),
          child: AppBar(
            flexibleSpace: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(0.5),
                    Colors.orangeAccent,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
            title: const Text(
              "แก้ไขเมนู",
              style: TextStyle(
                fontFamily: "assets/fonts/ChakraPetch-Bold.ttf",
                color: Color.fromARGB(255, 0, 0, 0),
                fontWeight: FontWeight.bold,
              ),
            ),
            centerTitle: true,
            elevation: 5,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(
                labelText: "ชื่อ",
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 20),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(
                labelText: "ประเภท",
                border: OutlineInputBorder(),
              ),
              value: selectedCategory,
              items: [
                'ผัด',
                'ต้ม',
                'ทอด',
                'ยำ / ลาบ',
                'แกง',
                'ส้มตำ',
              ]
                  .map((category) => DropdownMenuItem(
                        value: category,
                        child: Text(category),
                      ))
                  .toList(),
              onChanged: (value) {
                setState(() {
                  selectedCategory = value;
                });
              },
            ),
            const SizedBox(height: 20),
            TextField(
              controller: priceController,
              decoration: const InputDecoration(
                labelText: "ราคา",
                border: OutlineInputBorder(),
                suffixText: "THB",
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 20),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(
                labelText: "เนื้อสัตว์",
                border: OutlineInputBorder(),
              ),
              value: selectedDetails,
              items: [
                'หมู',
                'ไก่',
                'เนื้อ',
                'กุ้ง',
                'ปลาหมึก',
                'ทะเลรวม',
                'รวมมิตร',
                'อื่นๆ'
              ]
                  .map((detail) => DropdownMenuItem(
                        value: detail,
                        child: Text(detail),
                      ))
                  .toList(),
              onChanged: (value) {
                setState(() {
                  selectedDetails = value;
                });
              },
            ),
            const Spacer(),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                    ),
                    child: const Text(
                      "ยกเลิก",
                      style: TextStyle(
                        color: Colors.white, // เปลี่ยนสีฟอนต์เป็นสีขาว
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton(
                    onPressed: updateItem,
                    style:
                        ElevatedButton.styleFrom(backgroundColor: Colors.green),
                    child: const Text(
                      "บันทึก",
                      style: TextStyle(
                        color: Colors.white, // เปลี่ยนสีฟอนต์เป็นสีขาว
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
