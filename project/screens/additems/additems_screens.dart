import 'dart:typed_data';
import 'dart:convert';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/screens/base64/base64_image_picker.dart';

class AddItemsScreen extends StatefulWidget {
  const AddItemsScreen({Key? key}) : super(key: key);

  @override
  _AddItemsScreenState createState() => _AddItemsScreenState();
}

class _AddItemsScreenState extends State<AddItemsScreen> {
  String? selectedCategory;
  TextEditingController nameController = TextEditingController();
  TextEditingController priceController = TextEditingController();
  String? selectedDetails;
  Uint8List? _imageBytes; // ใช้ Uint8List แทน File

  // ฟังก์ชันเลือกรูปจากหน้า Base64ImagePicker
  Future<void> openImagePickerScreen() async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => Base64ImagePicker()),
    );

    if (result != null && result is Uint8List) {
      setState(() {
        _imageBytes = result;
      });
    }
  }

  // แปลงรูปภาพเป็น Base64 String
  Future<String?> convertImageToBase64(Uint8List imageBytes) async {
    try {
      return base64Encode(imageBytes);
    } catch (e) {
      return null;
    }
  }

  Future<void> addItem() async {
    try {
      final String name = nameController.text.trim();
      final String price = priceController.text.trim();

      if (name.isEmpty ||
          price.isEmpty ||
          selectedCategory == null ||
          selectedDetails == null ||
          _imageBytes == null) {
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text("ข้อมูลไม่ครบถ้วน"),
            content: const Text("กรุณากรอกข้อมูลให้ครบถ้วนและเลือกรูปภาพ"),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text("ปิด"),
              ),
            ],
          ),
        );
        return;
      }

      // แปลงรูปเป็น Base64
      String? imageBase64 = await convertImageToBase64(_imageBytes!);
      if (imageBase64 == null) {
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text("เกิดข้อผิดพลาด"),
            content: const Text("ไม่สามารถแปลงรูปภาพได้"),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text("ปิด"),
              ),
            ],
          ),
        );
        return;
      }

      // บันทึกลง Firestore
      await FirebaseFirestore.instance.collection('Foods').add({
        'name': name,
        'category': selectedCategory,
        'price': price,
        'details': selectedDetails,
        'imageBase64': imageBase64, // บันทึกรูปเป็น Base64
        'isAvailable': true, // ✅ เพิ่มฟิลด์ isAvailable และตั้งค่าเป็น true
        'createdAt': FieldValue.serverTimestamp(),
      });

      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("เพิ่มเมนูสำเร็จ"),
          content: const Text("เมนูของคุณได้ถูกเพิ่มไปยังระบบแล้ว"),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                Navigator.pop(context);
              },
              child: const Text("ตกลง"),
            ),
          ],
        ),
      );
    } catch (e) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("เกิดข้อผิดพลาด"),
          content: Text("ไม่สามารถเพิ่มเมนูได้ในขณะนี้: $e"),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
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
              "เพิ่มเมนูอาหาร",
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
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            GestureDetector(
              onTap: openImagePickerScreen, // เปิดหน้าจอเลือกภาพ
              child: Container(
                height: 150,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: _imageBytes == null
                    ? const Center(child: Text("เพิ่มรูปภาพ"))
                    : Image.memory(_imageBytes!,
                        fit: BoxFit.cover), // แสดงรูปที่เลือก
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: nameController,
              decoration: const InputDecoration(
                labelText: "ชื่อเมนู",
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            StreamBuilder<QuerySnapshot>(
              stream: FirebaseFirestore.instance
                  .collection('Categories')
                  .snapshots(),
              builder: (context, snapshot) {
                if (!snapshot.hasData) {
                  return CircularProgressIndicator(); // ✅ แสดงโหลดข้อมูลระหว่างดึงข้อมูล
                }

                var categoryDocs = snapshot.data!.docs;

                return DropdownButtonFormField<String>(
                  decoration: const InputDecoration(
                    labelText: "ประเภทเมนู",
                    border: OutlineInputBorder(),
                  ),
                  value: selectedCategory, // ค่าที่เลือกไว้
                  items: categoryDocs.map((doc) {
                    return DropdownMenuItem<String>(
                      value: doc['name'],
                      child: Text(doc['name']),
                    );
                  }).toList(),
                  onChanged: (value) {
                    setState(() {
                      selectedCategory = value; // ✅ อัปเดตค่าที่เลือก
                    });
                  },
                );
              },
            ),
            const SizedBox(height: 16),
            TextField(
              controller: priceController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: "ราคา",
                border: OutlineInputBorder(),
                suffixText: "บาท",
              ),
            ),
            const SizedBox(height: 16),
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
                'อื่นๆ',
                '-'
              ]
                  .map((detail) =>
                      DropdownMenuItem(value: detail, child: Text(detail)))
                  .toList(),
              onChanged: (value) {
                setState(() {
                  selectedDetails = value;
                });
              },
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red.shade400,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0)),
                    ),
                    child: const Text("ยกเลิก",
                        style: TextStyle(color: Colors.white)),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: addItem,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green.shade400,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0)),
                    ),
                    child: const Text("บันทึก",
                        style: TextStyle(color: Colors.white)),
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
