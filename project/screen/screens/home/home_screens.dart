import 'dart:convert';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:krua_pa_ree/screens/login/login_screens.dart';
import 'package:krua_pa_ree/screens/menusearchbelegate/MenuSearchDelegate.dart';
import 'package:krua_pa_ree/screens/orderhistory/orderhistory_screen.dart';
import 'package:krua_pa_ree/screens/profile/editprofile_screen.dart';
import '../Orderuse/Orderuse_screens.dart';
import '../cart/cart_screen.dart';
import '../homeowner/editpassword_screen.dart';
import '../menudetail/menudetail_screens.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String searchQuery = '';
  TextEditingController searchController = TextEditingController();

  void showLogoutConfirmationDialog(BuildContext context) {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยกดข้างนอก
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20), // ขอบโค้งมน
          ),
          title: Row(
            children: const [
              Icon(Icons.logout, color: Colors.red, size: 28),
              SizedBox(width: 8),
              Text(
                "ยืนยันการออกจากระบบ",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          content: const Text(
            "คุณแน่ใจหรือไม่ว่าต้องการออกจากระบบ?",
            style: TextStyle(fontSize: 16),
          ),
          actionsAlignment: MainAxisAlignment.spaceBetween,
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext); // ปิดป็อปอัป
              },
              child: const Text(
                "ยกเลิก",
                style:
                    TextStyle(color: Colors.grey, fontWeight: FontWeight.bold),
              ),
            ),
            ElevatedButton(
              onPressed: () async {
                try {
                  await FirebaseAuth.instance.signOut(); // ออกจากระบบ
                  Navigator.pop(dialogContext); // ปิดป็อปอัป

                  // นำไปหน้า Login หลังออกจากระบบ
                  Navigator.pushAndRemoveUntil(
                    context,
                    MaterialPageRoute(builder: (context) => LoginScreen()),
                    (route) => false,
                  );
                } catch (e) {
                  Navigator.pop(dialogContext); // ปิดป็อปอัป
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text("เกิดข้อผิดพลาด: $e"),
                      backgroundColor: Colors.red,
                      behavior: SnackBarBehavior.floating,
                    ),
                  );
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: const Text(
                "ยืนยัน",
                style:
                    TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    if (user == null) {
      return Scaffold(
        body: Center(child: Text("ยังไม่มีผู้ใช้ล็อกอิน")),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'KRUA PA REE',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(Icons.search),
            onPressed: () {
              showSearch(
                context: context,
                delegate: MenuSearchDelegate(),
              );
            },
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          children: [
            FutureBuilder<DocumentSnapshot>(
              future: FirebaseFirestore.instance
                  .collection('Customers')
                  .doc(user.uid)
                  .get(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (snapshot.hasError ||
                    !snapshot.hasData ||
                    !snapshot.data!.exists) {
                  return const Center(child: Text('ไม่พบข้อมูลผู้ใช้'));
                }

                final userData = snapshot.data!.data() as Map<String, dynamic>;
                final name = userData['name'] ?? "ไม่พบชื่อผู้ใช้";

                return UserAccountsDrawerHeader(
                  accountName: Text(name),
                  accountEmail: Text(user.email ?? "ไม่พบอีเมล"),
                  decoration: const BoxDecoration(
                      color: Color.fromARGB(255, 214, 141, 58)),
                  currentAccountPicture: CircleAvatar(
                    backgroundColor: Colors.white,
                    child: Icon(Icons.person, color: Colors.blue),
                  ),
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.home),
              title: Text("หน้าหลัก"),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(Icons.edit),
              title: Text("แก้ไขข้อมูลส่วนตัว"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) =>
                          EditProfileScreen(userId: user.uid)),
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.edit),
              title: Text("แก้ไขรหัสผ่าน"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => EditPasswordScreen()),
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.history),
              title: Text("ประวัติคำสั่งซื้อ"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => OrderHistoryScreen()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.logout),
              title: const Text("ออกจากระบบ"),
              onTap: () {
                showLogoutConfirmationDialog(
                    context); // เรียกฟังก์ชันแสดงป็อปอัป
              },
            ),
          ],
        ),
      ),
      body: Container(
        width: double.infinity, // ให้ครอบคลุมความกว้างทั้งหมด
        height: double.infinity, // ให้ครอบคลุมความสูงทั้งหมด
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Column(
          children: [
            FutureBuilder<QuerySnapshot>(
              future: FirebaseFirestore.instance
                  .collection('Orders')
                  .where('userId', isEqualTo: user.uid)
                  .get(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasError) {
                  return Center(
                      child: Text("เกิดข้อผิดพลาด: ${snapshot.error}"));
                }

                if (snapshot.data != null && snapshot.data!.docs.isNotEmpty) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 10.0),
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => OrderUseScreen(),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        padding: const EdgeInsets.symmetric(
                            vertical: 10, horizontal: 20),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      icon: Icon(Icons.shopping_bag, color: Colors.white),
                      label: Text(
                        "ตรวจสอบสถานะคำสั่งซื้อ",
                        style: TextStyle(fontSize: 16, color: Colors.white),
                      ),
                    ),
                  );
                }
                return SizedBox.shrink();
              },
            ),
            Expanded(
              child: StreamBuilder<QuerySnapshot>(
                stream: getFoodMenu(),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return Center(child: CircularProgressIndicator());
                  }
                  if (snapshot.hasError) {
                    return Center(
                        child: Text("เกิดข้อผิดพลาด: ${snapshot.error}"));
                  }

                  if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                    return Center(child: Text("ไม่มีเมนูอาหาร"));
                  }

                  // ดึงข้อมูลเมนูอาหาร
                  final menuItems = snapshot.data!.docs.map((doc) {
                    final data = doc.data() as Map<String, dynamic>;
                    return {
                      "name": data['name'] ?? "ไม่มีชื่อเมนู",
                      "imageBase64": data['imageBase64'] ?? "",
                      "price": data['price']?.toString() ?? "0",
                      "category": data['category'] ?? "ไม่ระบุหมวดหมู่",
                      "isAvailable": data['isAvailable'] ??
                          true, // ✅ กำหนดค่า Default เป็น true
                    };
                  }).toList();

                  // กรองข้อมูลตามข้อความค้นหา
                  final filteredItems = menuItems.where((item) {
                    return item["name"]
                        .toLowerCase()
                        .contains(searchQuery.toLowerCase());
                  }).toList();

                  return GridView.builder(
                    padding: const EdgeInsets.all(16.0),
                    itemCount: filteredItems.length,
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      crossAxisSpacing: 10,
                      mainAxisSpacing: 10,
                      childAspectRatio: 0.9,
                    ),
                    itemBuilder: (context, index) {
                      final menuItem = filteredItems[index];
                      return GestureDetector(
                        onTap: menuItem["isAvailable"]
                            ? () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => MenuDetailScreen(
                                      name: menuItem["name"],
                                      image: menuItem["imageBase64"],
                                      price: menuItem["price"],
                                      category: menuItem[
                                          "category"], // ส่ง category ที่ดึงจาก Firestore
                                    ),
                                  ),
                                );
                              }
                            : null, // ❌ ปิดการกดเลือกเมนูที่หมด
                        child: Opacity(
                          opacity: menuItem["isAvailable"]
                              ? 1.0
                              : 0.5, // เมนูหมดแสดงจางลง
                          child: _menuCard(
                            menuItem["imageBase64"],
                            menuItem["name"],
                            menuItem["price"],
                          ),
                        ),
                      );
                    },
                  );
                },
              ),
            )
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
  backgroundColor: Colors.orange,
  onPressed: () async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => CartScreen()),
    );

    // ✅ รีเฟรชหน้า HomeScreen เมื่อกลับมา
    setState(() {});
  },
  child: Icon(Icons.shopping_cart, color: Colors.white),
),

    );
  }

  Stream<QuerySnapshot> getFoodMenu() {
    return FirebaseFirestore.instance.collection('Foods').snapshots();
  }

  Widget _menuCard(String base64Image, String title, String price) {
    Uint8List? imageBytes;

    try {
      if (base64Image.isNotEmpty) {
        imageBytes = base64Decode(base64Image);
      }
    } catch (e) {
      print("Error decoding image: $e");
    }

    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      elevation: 5,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(15),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: imageBytes != null
                  ? Image.memory(
                      imageBytes,
                      fit: BoxFit.cover,
                    )
                  : const Icon(
                      Icons.broken_image,
                      size: 100,
                      color: Colors.grey,
                    ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    "$price บาท",
                    style: const TextStyle(fontSize: 14, color: Colors.green),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

//ค้นหา
// class MenuSearchDelegate extends SearchDelegate {
//   @override
//   List<Widget> buildActions(BuildContext context) {
//     return [
//       IconButton(
//         icon: Icon(Icons.clear),
//         onPressed: () {
//           query = '';
//         },
//       ),
//     ];
//   }

//   @override
//   Widget buildLeading(BuildContext context) {
//     return IconButton(
//       icon: Icon(Icons.arrow_back),
//       onPressed: () {
//         close(context, null);
//       },
//     );
//   }

//   @override
//   Widget buildResults(BuildContext context) {
//     return Container();
//   }

//   @override
//   Widget buildSuggestions(BuildContext context) {
//     return Container();
//   }
// }
