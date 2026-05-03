import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class EditPasswordScreen extends StatelessWidget {
  EditPasswordScreen({Key? key}) : super(key: key);

  final TextEditingController oldPasswordController = TextEditingController();
  final TextEditingController newPasswordController = TextEditingController();
  final TextEditingController confirmPasswordController = TextEditingController();

  // 🔹 ฟังก์ชันแสดงป็อปอัปแจ้งเตือน (ภาษาไทย)
  void _showAlertDialog(BuildContext context, String title, String message, {bool isError = false}) {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยคลิกข้างนอก
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          contentPadding: const EdgeInsets.all(16),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                isError ? Icons.error_outline : Icons.check_circle,
                color: isError ? Colors.red : Colors.green,
                size: 50,
              ),
              const SizedBox(height: 16),
              Text(
                title,
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Text(
                message,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context); // ปิดป็อปอัป
                  if (!isError) {
                    Navigator.pushReplacementNamed(context, '/homeowner_screen'); // กลับไปหน้าเจ้าของร้าน
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: isError ? Colors.redAccent : Colors.green,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                child: const Text(
                  "ตกลง",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> updatePassword(BuildContext context) async {
    try {
      User? user = FirebaseAuth.instance.currentUser;

      if (user == null) {
        _showAlertDialog(context, "ข้อผิดพลาด", "ไม่พบข้อมูลผู้ใช้ กรุณาเข้าสู่ระบบใหม่", isError: true);
        return;
      }

      if (newPasswordController.text.isEmpty || confirmPasswordController.text.isEmpty) {
        _showAlertDialog(context, "กรุณากรอกข้อมูลให้ครบ", "กรุณากรอกข้อมูลรหัสผ่านใหม่และยืนยันรหัสผ่านให้ครบถ้วน", isError: true);
        return;
      }

      if (newPasswordController.text.length < 6) {
        _showAlertDialog(context, "รหัสผ่านสั้นเกินไป", "รหัสผ่านต้องไม่น้อยกว่า 6 ตัวอักษร", isError: true);
        return;
      }

      if (newPasswordController.text != confirmPasswordController.text) {
        _showAlertDialog(context, "รหัสผ่านไม่ตรงกัน", "กรุณากรอกรหัสผ่านใหม่ให้ตรงกัน", isError: true);
        return;
      }

      String email = user.email ?? "";
      AuthCredential credential = EmailAuthProvider.credential(
        email: email,
        password: oldPasswordController.text,
      );

      await user.reauthenticateWithCredential(credential);
      await user.updatePassword(newPasswordController.text);

      await FirebaseFirestore.instance.collection('PasswordChanges').add({
        'userId': user.uid,
        'updatedAt': FieldValue.serverTimestamp(),
      });

      _showAlertDialog(context, "เปลี่ยนรหัสผ่านสำเร็จ", "รหัสผ่านของคุณได้รับการอัปเดตเรียบร้อยแล้ว!");
    } catch (e) {
      _showAlertDialog(context, "เกิดข้อผิดพลาด", "ไม่สามารถเปลี่ยนรหัสผ่านได้ กรุณาลองใหม่: $e", isError: true);
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
                  colors: [Colors.orange.withOpacity(0.5), Colors.orangeAccent],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
            title: const Text(
              "แก้ไขรหัสผ่าน",
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
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                const SizedBox(height: 16),
                Image.asset(
                  'assets/images/krua pa ree.png', // ใส่โลโก้ร้าน
                  height: 155,
                ),
                const SizedBox(height: 16),
                _buildPasswordField("รหัสผ่านเก่า", oldPasswordController),
                const SizedBox(height: 16),
                _buildPasswordField("รหัสผ่านใหม่", newPasswordController),
                const SizedBox(height: 16),
                _buildPasswordField("ยืนยันรหัสผ่านใหม่", confirmPasswordController),
                const SizedBox(height: 32),
                ElevatedButton(
                  onPressed: () async {
                    await updatePassword(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 80),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                  ),
                  child: const Text(
                    "บันทึก",
                    style: TextStyle(fontSize: 18, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPasswordField(String label, TextEditingController controller) {
    return TextField(
      controller: controller,
      obscureText: true,
      decoration: InputDecoration(
        labelText: label,
        filled: true,
        fillColor: Colors.orange.withOpacity(0.2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
    );
  }
}
