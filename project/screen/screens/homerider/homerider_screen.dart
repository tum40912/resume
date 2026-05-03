import 'package:cloud_firestore/cloud_firestore.dart'; // ต้องเพิ่ม Firestore
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/screens/orderdetail/orderdetail_Screen.dart';

import '../homeowner/editpassword_screen.dart';
import '../login/login_screens.dart';

void main() {
  runApp(HomeRiderScreen());
}

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
              style: TextStyle(color: Colors.grey, fontWeight: FontWeight.bold),
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

class HomeRiderScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.orange,
        fontFamily: 'Roboto',
      ),
      home: InputPage(),
    );
  }
}

class InputPage extends StatefulWidget {
  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  User? currentUser;
  String? role;

  @override
  void initState() {
    super.initState();
    _fetchUserData();
  }

  Future<void> _fetchUserData() async {
    try {
      User? user = FirebaseAuth.instance.currentUser;
      setState(() {
        currentUser = user;
      });

      if (user != null) {
        DocumentSnapshot userDoc = await FirebaseFirestore.instance
            .collection('Employees')
            .doc(user.uid)
            .get();

        if (userDoc.exists) {
          String userRole = userDoc['role'];
          if (userRole == "rider") {
            setState(() {
              role = userRole;
            });
          } else {
            setState(() {
              role = null;
            });
            _showErrorDialog("บัญชีของคุณไม่มีสิทธิ์เข้าถึงข้อมูลนี้");
          }
        }
      }
    } catch (e) {
      print('เกิดข้อผิดพลาด: $e');
    }
  }

  Stream<QuerySnapshot> getCompletedOrders() {
    return FirebaseFirestore.instance
        .collection('Orders')
        .where('status', isEqualTo: 'Completed')
        .snapshots();
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("เกิดข้อผิดพลาด"),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text("ปิด"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60), // กำหนดความสูงของ AppBar
        child: ClipRRect(
          borderRadius: const BorderRadius.only(
            bottomLeft: Radius.circular(20), // ขอบโค้งมนด้านซ้ายล่าง
            bottomRight: Radius.circular(20), // ขอบโค้งมนด้านขวาล่าง
          ),
          child: AppBar(
            flexibleSpace: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(0.5), // สีส้มไล่เฉด
                    Colors.orangeAccent,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
            title: const Text(
              "รายการอาหารที่ต้องจัดส่ง",
              style: TextStyle(
                fontFamily: "assets/fonts/ChakraPetch-Bold.ttf",
                color: Color.fromARGB(255, 0, 0, 0),
                fontWeight: FontWeight.bold,
              ),
            ),
            centerTitle: true, // จัดกึ่งกลางข้อความ
            elevation: 5, // เพิ่มเงา
          ),
        ),
      ),
      drawer: Drawer(
        child: Column(
          children: [
            UserAccountsDrawerHeader(
              accountName: Text(role == "rider" ? "Rider" : "ไม่มีสิทธิ์"),
              accountEmail: Text(currentUser?.email ?? "ไม่พบอีเมล"),
              decoration:
                  const BoxDecoration(color: Color.fromARGB(255, 193, 83, 236)),
              currentAccountPicture: const CircleAvatar(
                backgroundColor: Colors.white,
                child: Icon(Icons.person, color: Colors.orange, size: 50),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.home),
              title: const Text("หน้าหลัก"),
              onTap: () => Navigator.pop(context),
            ),
            ListTile(
              leading: const Icon(Icons.edit),
              title: const Text("แก้ไขรหัสผ่าน"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => EditPasswordScreen()),
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
        width: double.infinity, // กำหนดความกว้างให้เต็มจอ
        height: double.infinity, // กำหนดความสูงให้เต็มจอ
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color.fromARGB(255, 240, 239, 237)
                  .withOpacity(0.5), // สีไล่เฉด
              const Color.fromARGB(255, 252, 213, 162),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: StreamBuilder<QuerySnapshot>(
            stream: getCompletedOrders(),
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(child: CircularProgressIndicator());
              }
              if (snapshot.hasError) {
                return const Center(
                    child: Text("เกิดข้อผิดพลาดในการดึงข้อมูล"));
              }
              if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                return const Center(
                    child: Text("ไม่มีรายการอาหารที่ต้องจัดส่ง"));
              }

              final orders = snapshot.data!.docs;

              return ListView.builder(
                itemCount: orders.length,
                itemBuilder: (context, index) {
                  final orderData =
                      orders[index].data() as Map<String, dynamic>;
                  final orderId =
                      orders[index].id; // ✅ ดึง orderId จาก Firestore
                  final customerId = orderData['userId'] ?? 'ไม่มีข้อมูล';

                  return FutureBuilder<DocumentSnapshot>(
                    future: FirebaseFirestore.instance
                        .collection('Customers')
                        .doc(customerId)
                        .get(),
                    builder: (context, customerSnapshot) {
                      if (customerSnapshot.connectionState ==
                          ConnectionState.waiting) {
                        return const Center(child: CircularProgressIndicator());
                      }

                      if (!customerSnapshot.hasData ||
                          !customerSnapshot.data!.exists) {
                        return Card(
                          child: ListTile(
                            title: Text(
                                "ไม่พบข้อมูลลูกค้า (Order ID: $orderId)"), // แสดง Order ID
                          ),
                        );
                      }

                      final customerData =
                          customerSnapshot.data!.data() as Map<String, dynamic>;

                      return Card(
                        elevation: 2,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.all(16),
                          title: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(Icons.person,
                                      color: Colors.blue,
                                      size: 24), // ✅ ไอคอนลูกค้า
                                  SizedBox(width: 8),
                                  Text(
                                    customerData['name'],
                                    style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.black),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 5),
                              Row(
                                children: [
                                  Icon(Icons.badge,
                                      color: Colors.orange,
                                      size: 24), // ✅ ไอคอน Order ID
                                  SizedBox(width: 8),
                                  Text(
                                    "Order ID: $orderId",
                                    style: const TextStyle(
                                        fontSize: 14, color: Colors.grey),
                                  ),
                                ],
                              ),
                            ],
                          ),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Icon(Icons.location_on,
                                      color: Colors.red,
                                      size: 24), // ✅ ไอคอนที่อยู่
                                  SizedBox(width: 8),
                                  Expanded(
                                    // ✅ ป้องกันข้อความที่อยู่ล้น
                                    child: Text(
                                      customerData['address'],
                                      style: const TextStyle(fontSize: 14),
                                      softWrap: true, // ✅ รองรับข้อความยาว
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 5),
                              Text(
                                "ยอดรวม: ${orderData['totalPrice']} บาท",
                                style: const TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.green),
                              ),
                            ],
                          ),
                          trailing: const Icon(Icons.arrow_forward_ios,
                              color: Colors.orange),
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => OrderDetailScreen(
                                  order: orderData,
                                  customerName: customerData['name'],
                                  customerAddress: customerData['address'],
                                  customerPhone: customerData['phone'],
                                  orderId: orderId, // ✅ ส่ง Order ID ไปด้วย
                                ),
                              ),
                            );
                          },
                        ),
                      );
                    },
                  );
                },
              );
            },
          ),
        ),
      ),
    );
  }
}
