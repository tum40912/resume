import 'dart:convert';
import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../home/home_screens.dart';

class MenuDetailScreen extends StatefulWidget {
  final String name;
  final String image;
  final String price;
  final String category; // กำหนดว่า category ต้องไม่เป็น null

  MenuDetailScreen({
    required this.name,
    required this.image,
    required this.price,
    this.category = "ไม่ระบุหมวดหมู่", // กำหนดค่าเริ่มต้น
  });

  @override
  _MenuDetailScreenState createState() => _MenuDetailScreenState();
}

class _MenuDetailScreenState extends State<MenuDetailScreen> {
  int quantity = 1;
  List<Map<String, dynamic>> selectedSides = [];
  List<Map<String, dynamic>> selectedSpecials = []; // ตัวเลือกพิเศษ
  final TextEditingController noteController = TextEditingController();

  final List<Map<String, dynamic>> sides = [
    {'name': 'ไข่ดาว', 'price': 10},
    {'name': 'ไข่เจียว', 'price': 20},
    {'name': 'ไข่ข้น', 'price': 15},
    {'name': 'กุนเชียงหมู', 'price': 15},
    {'name': 'พิเศษข้าว', 'price': 10},
    {'name': 'พิเศษเนื้อสัตว์', 'price': 10},
  ];

  Uint8List? _imageBytes;

  @override
  void initState() {
    super.initState();
    _convertImage();
  }

  void _convertImage() {
    try {
      if (widget.image.isNotEmpty) {
        _imageBytes = base64Decode(widget.image); // แปลง Base64 เป็น Uint8List
      }
    } catch (e) {
      print("Error decoding Base64 image: $e");
    }
  }

  Future<void> _confirmAddToCart() async {
    showDialog(
      context: context,
      barrierDismissible: false, // ต้องกดปุ่มเท่านั้นถึงจะปิด
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Row(
            children: [
              Icon(Icons.shopping_cart, color: Colors.green, size: 30),
              SizedBox(width: 10),
              Text("ยืนยันการสั่งซื้อ"),
            ],
          ),
          content: Text("คุณต้องการเพิ่ม '${widget.name}' ลงในตะกร้าหรือไม่?"),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext); // ปิดป๊อปอัป
              },
              child: Text("ยกเลิก", style: TextStyle(color: Colors.grey)),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
              onPressed: () async {
                Navigator.pop(dialogContext); // ปิดป๊อปอัป
                _addToCart(); // ✅ เพิ่มลงตะกร้า
              },
              child: Text("ยืนยัน", style: TextStyle(color: Colors.white)),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.white, Color(0xFFF4E1D2)],
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
      ),
      child: Scaffold(
        backgroundColor: Colors.transparent,
        appBar: AppBar(
          title: Text(
            widget.name,
            style: const TextStyle(
              color: Colors.black87,
              fontWeight: FontWeight.bold,
            ),
          ),
          backgroundColor: Colors.white,
          foregroundColor: Colors.black,
          elevation: 5,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(
              bottom: Radius.circular(20),
            ),
          ),
        ),
        body: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildImageSection(),
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.name,
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      "${widget.price} บาท",
                      style: const TextStyle(
                        fontSize: 22,
                        color: Colors.green,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    // const SizedBox(height: 16),
                    // _buildSpecialSection(), // ✅ เรียกใช้ส่วนตัวเลือกพิเศษ
                    const SizedBox(height: 16),
                    _buildSidesSection(), // ✅ ต้องแน่ใจว่าเรียกใช้ที่นี่
                    const SizedBox(height: 16),
                    _buildNoteSection(),
                    const SizedBox(height: 16),
                    _buildQuantitySection(),
                    const SizedBox(height: 16),
                    Center(
                      child: ElevatedButton(
                        onPressed:
                            _confirmAddToCart, // ✅ ใช้ป๊อปอัปก่อนเพิ่มลงตะกร้า
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFFF7043),
                          padding: const EdgeInsets.symmetric(
                              vertical: 15, horizontal: 50),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30)),
                          elevation: 5,
                        ),
                        child: const Text(
                          "เพิ่มไปยังตะกร้า",
                          style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.white),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// **🔹 ฟังก์ชันแสดงภาพ (ใช้ได้ทั้ง Base64 & URL)**
  Widget _buildImageSection() {
    return ClipRRect(
      borderRadius: const BorderRadius.only(
        bottomLeft: Radius.circular(30),
        bottomRight: Radius.circular(30),
      ),
      child: _imageBytes != null
          ? Image.memory(
              _imageBytes!,
              height: 250,
              width: double.infinity,
              fit: BoxFit.cover,
            )
          : Image.network(
              widget.image,
              height: 250,
              width: double.infinity,
              fit: BoxFit.cover,
              errorBuilder: (context, error, stackTrace) {
                return const Icon(Icons.broken_image,
                    size: 100, color: Colors.grey);
              },
            ),
    );
  }

  // ฟังก์ชันการแสดงตัวเลือกเครื่องเคียง
  Widget _buildSidesSection() {
    // ตรวจสอบค่าของ category ก่อนเปรียบเทียบ
    if ((widget.category ?? "").toLowerCase() == "เครื่องดื่ม") {
      return SizedBox.shrink(); // ไม่แสดง widget
    }

    // ถ้าไม่ใช่ "เครื่องดื่ม" ให้แสดงตัวเลือกเพิ่มเติม
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "เพิ่มเติม (ไม่จำกัดตัวเลือก)",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Column(
              children: sides.map((side) {
                return CheckboxListTile(
                  activeColor: Colors.orange,
                  title: Text(
                    "${side['name']} (+${side['price']} บาท)",
                    style: const TextStyle(fontSize: 16),
                  ),
                  value:
                      selectedSides.any((item) => item['name'] == side['name']),
                  onChanged: (bool? selected) {
                    setState(() {
                      if (selected == true) {
                        selectedSides.add(side);
                      } else {
                        selectedSides.removeWhere(
                            (item) => item['name'] == side['name']);
                      }
                    });
                  },
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  // ฟังก์ชันการแสดงหมายเหตุ
  Widget _buildNoteSection() {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "หมายเหตุถึงร้านอาหาร",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: noteController,
              decoration: InputDecoration(
                hintText: "ระบุรายละเอียดคำขอ",
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              maxLines: 3,
            ),
          ],
        ),
      ),
    );
  }

  // ฟังก์ชันการแสดงปริมาณ
  Widget _buildQuantitySection() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            IconButton(
              onPressed: () {
                if (quantity > 1) {
                  setState(() {
                    quantity--;
                  });
                }
              },
              icon: const Icon(Icons.remove_circle_outline,
                  color: Colors.redAccent),
            ),
            Text(
              quantity.toString(),
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            IconButton(
              onPressed: () {
                setState(() {
                  quantity++;
                });
              },
              icon: const Icon(Icons.add_circle_outline, color: Colors.green),
            ),
          ],
        ),
        Text(
          "รวม ${_calculateTotalPrice()} บาท",
          style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Color.fromARGB(255, 21, 184, 75)),
        ),
      ],
    );
  }

  // ฟังก์ชันคำนวณราคาสุทธิ
  int _calculateTotalPrice() {
    int basePrice = int.parse(widget.price);
    int sidesPrice =
        selectedSides.fold(0, (total, side) => total + (side['price'] as int));
    int specialPrice = selectedSpecials.fold(
        0, (total, special) => total + (special['price'] as int));
    return (basePrice + sidesPrice + specialPrice) * quantity;
  }

  // ฟังก์ชันเพิ่มไปยังตะกร้า
  void _addToCart() {
    FirebaseFirestore.instance.collection('Cart').add({
      'name': widget.name,
      'image': widget.image,
      'price': widget.price,
      'userId': FirebaseAuth.instance.currentUser?.uid,
      'quantity': quantity,
      'sides': selectedSides,
      'specials': selectedSpecials, // เพิ่มตัวเลือกพิเศษลงในตะกร้า
      'note': noteController.text,
      'totalPrice': _calculateTotalPrice(),
    }).then((value) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20), // ปรับขอบโค้งมน
          ),
          title: Column(
            children: [
              Icon(Icons.check_circle,
                  color: Colors.green, size: 60), // ไอคอน ✔️
              SizedBox(height: 10),
              Text(
                "เพิ่มสินค้าสำเร็จ",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.green[800], // สีเขียวเข้ม
                ),
              ),
            ],
          ),
          content: Text(
            'เพิ่ม "${widget.name}" ใส่ตะกร้าเรียบร้อยแล้ว',
            textAlign: TextAlign.center, // จัดข้อความตรงกลาง
            style: TextStyle(fontSize: 16, color: Colors.black87),
          ),
          actions: [
            Center(
              // จัดปุ่มให้อยู่ตรงกลาง
              child: ElevatedButton(
                onPressed: () {
                  // ไปที่หน้า HomeScreen เมื่อกดปุ่ม "ตกลง"
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (context) => HomeScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green, // ปรับสีปุ่ม
                  padding: EdgeInsets.symmetric(vertical: 12, horizontal: 30),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10), // ปรับขอบมน
                  ),
                ),
                child: Text(
                  "ตกลง",
                  style: TextStyle(fontSize: 16, color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      );
    }).catchError((error) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('เกิดข้อผิดพลาด: $error')),
      );
    });
  }
}
