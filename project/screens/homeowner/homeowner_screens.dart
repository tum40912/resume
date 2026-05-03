import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/payment/PaymentVerificationScreen.dart';
import 'package:krua_pa_ree/screens/foodmenuowner/foodmenuowner_screens.dart';
import 'package:krua_pa_ree/screens/order/order_screens.dart';
import 'package:krua_pa_ree/screens/addemp/addemp_screen.dart';

import '../login/login_screens.dart';
import '../report/report_screens.dart';
import 'editpassword_screen.dart';

class HomeOwnerScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser; // ดึงข้อมูลผู้ใช้ปัจจุบัน

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
            centerTitle: true, // จัดกึ่งกลางข้อความ
            elevation: 5, // เพิ่มเงา
          ),
        ),
      ),
      drawer: Drawer(
        child: ListView(
          children: [
            UserAccountsDrawerHeader(
              accountName: Text("เจ้าของร้าน"),
              accountEmail: Text(user?.email ?? "ไม่พบอีเมล"),
              currentAccountPicture: CircleAvatar(
                backgroundColor: Colors.white,
                child: Icon(Icons.person, color: Colors.blue),
              ),
            ),
            ListTile(
              leading: Icon(Icons.home),
              title: Text("หน้าหลัก"),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(Icons.person_add), // เพิ่มไอคอนสำหรับเพิ่มผู้ใช้
              title: Text("เพิ่มบัญชีผู้ใช้"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) =>
                          AddEmpScreen()), // เชื่อมไปยังหน้า AddEmpPage
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.edit), // เพิ่มไอคอนสำหรับเพิ่มผู้ใช้
              title: Text("แก้ไขรหัสผ่าน"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) =>
                          EditPasswordScreen()), // เชื่อมไปยังหน้า AddEmpPage
                );
              },
            ),
            ListTile(
              leading: Icon(Icons.attach_money),
              title: Text("ตรวจสอบการชำระเงิน"),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => PaymentVerificationScreen()),
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
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              // เพิ่มโลโก้ด้านบนปุ่ม
              Container(
                margin: const EdgeInsets.only(
                    bottom: 20), // เพิ่มระยะห่างจากปุ่มด้านล่าง
                child: Column(
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Image.asset(
                          'assets/images/krua pa ree.png', // ใส่โลโก้ของร้าน
                          height: 200 // กำหนดขนาดโลโก้
                          ),
                    ),
                    const SizedBox(height: 10),
                  ],
                ),
              ),
              FractionallySizedBox(
                widthFactor: 0.75, // ตั้งให้กว้าง 70% ของหน้าจอ
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => OrderScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  icon: Icon(Icons.restaurant_menu),
                  label: Text(
                    "ออเดอร์",
                    style: TextStyle(fontSize: 18),
                  ),
                ),
              ),

              const SizedBox(height: 20),
              SizedBox(
                width: MediaQuery.of(context).size.width *
                    0.7, // กำหนดความกว้างเป็น 70% ของหน้าจอ
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => FoodMenumenuScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  icon: Icon(Icons.add),
                  label: Text(
                    "เพิ่มรายการอาหาร",
                    style: TextStyle(fontSize: 18),
                  ),
                ),
              ),

              const SizedBox(height: 20),
              FractionallySizedBox(
                widthFactor: 0.75, // กำหนดความกว้างเป็น 75% ของหน้าจอ
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ReportScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  icon: Icon(Icons.bar_chart),
                  label: Text(
                    "ดูรายงาน",
                    style: TextStyle(fontSize: 18),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
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
}
