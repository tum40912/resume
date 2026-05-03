import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/screens/login/login_screens.dart';

import '../../firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: RegisterScreen(),
    );
  }
}

class RegisterScreen extends StatefulWidget {
  @override
  _RegisterScreenState createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmPasswordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  void _showPasswordMismatchDialog() {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยการกดข้างนอก
      builder: (BuildContext context) {
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          contentPadding: const EdgeInsets.all(16),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.lock_outline, // ไอคอนล็อก
                color: Colors.red,
                size: 50,
              ),
              const SizedBox(height: 16),
              const Text(
                "รหัสผ่านไม่ตรงกัน",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              const Text(
                "กรุณากรอกรหัสผ่านให้ตรงกันก่อนดำเนินการต่อ",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context); // ปิดป็อปอัป
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.redAccent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                child: const Text(
                  "ตกลง",
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showRegistrationSuccessDialog() {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยกดข้างนอก
      builder: (BuildContext context) {
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          contentPadding: const EdgeInsets.all(16),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.check_circle, // ไอคอน ✔️
                color: Colors.green,
                size: 60,
              ),
              const SizedBox(height: 16),
              const Text(
                "สมัครสมาชิกสำเร็จ!",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              const Text(
                "บัญชีของคุณถูกสร้างเรียบร้อยแล้ว 🎉\nยินดีต้อนรับสู่แอปของเรา!",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context); // ปิดป็อปอัป
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(
                        builder: (context) =>
                            LoginScreen()), // ไปยังหน้าล็อกอิน
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 30, vertical: 12),
                ),
                child: const Text(
                  "เข้าสู่ระบบ",
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showErrorDialog(String errorMessage) {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยกดข้างนอก
      builder: (BuildContext context) {
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          contentPadding: const EdgeInsets.all(16),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.error_outline, // ไอคอนแจ้งเตือน ❗
                color: Colors.red,
                size: 50,
              ),
              const SizedBox(height: 16),
              const Text(
                "เกิดข้อผิดพลาด",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Text(
                errorMessage,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context); // ปิดป็อปอัป
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.redAccent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                child: const Text(
                  "ตกลง",
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> registerUser() async {
    if (passwordController.text != confirmPasswordController.text) {
      _showPasswordMismatchDialog(); // เรียกใช้ฟังก์ชันแสดงป็อปอัป
      return;
    }

    try {
      UserCredential userCredential =
          await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: emailController.text.trim(),
        password: passwordController.text.trim(),
      );

      await FirebaseFirestore.instance
          .collection('Customers')
          .doc(userCredential.user?.uid)
          .set({
        'email': emailController.text.trim(),
        'role': 'Customer',
        'createdAt': FieldValue.serverTimestamp(),
      });

      _showRegistrationSuccessDialog(); // เรียกใช้ฟังก์ชันแสดงป็อปอัป

      // เพิ่มการนำทางกลับไปยังหน้า LoginScreen
      Future.delayed(const Duration(seconds: 2), () {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => LoginScreen()),
        );
      });
    } on FirebaseAuthException catch (e) {
      String errorMessage;
      if (e.code == 'email-already-in-use') {
        errorMessage = "อีเมลนี้ได้รับการลงทะเบียนไปแล้ว";
      } else if (e.code == 'weak-password') {
        errorMessage = "รหัสผ่านต้องมีความยาวอย่างน้อย 6 ตัวอักษร";
      } else if (e.code == 'invalid-email') {
        errorMessage = "โปรดกรอกที่อยู่อีเมลที่ถูกต้อง";
      } else {
        errorMessage = e.message ?? "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ";
      }

      _showErrorDialog(errorMessage); // เรียกใช้ป็อปอัปแจ้งเตือน
    } catch (e) {
      _showErrorDialog("เกิดข้อผิดพลาดที่ไม่คาดคิด");
    }
  }

  void clearFields() {
    emailController.clear();
    passwordController.clear();
    confirmPasswordController.clear();
  }

  void showCenteredSnackbar(String message) {
    final overlay = Overlay.of(context);
    final overlayEntry = OverlayEntry(
      builder: (context) => Center(
        child: Material(
          color: Colors.transparent,
          child: Container(
            padding: const EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.8),
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: Text(
              message,
              style: const TextStyle(color: Colors.white, fontSize: 16),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );

    overlay?.insert(overlayEntry);

    Future.delayed(const Duration(seconds: 3))
        .then((_) => overlayEntry.remove());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity, // ทำให้ Container ขยายเต็มหน้าจอแนวนอน
        height: double.infinity, // ทำให้ Container ขยายเต็มหน้าจอแนวตั้ง
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Center(
          // ใช้ Center เพื่อให้เนื้อหาอยู่ตรงกลาง
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment:
                    MainAxisAlignment.center, // จัดเนื้อหาให้อยู่ตรงกลาง
                children: [
                  Image.asset(
                    'assets/images/krua pa ree.png',
                    height: 150, // ขนาดโลโก้
                  ),
                  const SizedBox(height: 20),
                  Card(
                    color: Colors.orange[100],
                    elevation: 8,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.0),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Center(
                              child: Text(
                                'สมัครสมาชิก',
                                style: TextStyle(
                                    fontSize: 24,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.black),
                              ),
                            ),
                            const SizedBox(height: 20),
                            _inputField('อีเมล', emailController,
                                isEmail: true),
                            const SizedBox(height: 16),
                            _inputField('รหัสผ่าน', passwordController,
                                obscureText: true),
                            const SizedBox(height: 16),
                            _inputField(
                                'ยืนยันรหัสผ่าน', confirmPasswordController,
                                obscureText: true),
                            const SizedBox(height: 20),
                            Center(
                              child: ElevatedButton(
                                onPressed: () {
                                  if (_formKey.currentState!.validate()) {
                                    registerUser();
                                  }
                                },
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.orange,
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 16, horizontal: 32),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                ),
                                child: const Text(
                                  'สมัครสมาชิก',
                                  style: TextStyle(
                                      fontSize: 18, color: Colors.white),
                                ),
                              ),
                            ),
                            const SizedBox(height: 10),
                            Center(
                              child: GestureDetector(
                                onTap: () {
                                  Navigator.pop(context);
                                },
                                child: const Text(
                                  'มีบัญชีผู้ใช้อยู่แล้ว?',
                                  style: TextStyle(
                                      fontSize: 14,
                                      color: Color.fromARGB(255, 56, 55, 55),
                                      decoration: TextDecoration.underline),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _inputField(String label, TextEditingController controller,
      {bool obscureText = false, bool isEmail = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 5),
        TextFormField(
          controller: controller,
          obscureText: obscureText,
          keyboardType:
              isEmail ? TextInputType.emailAddress : TextInputType.text,
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            contentPadding:
                const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
            filled: true,
            fillColor: Colors.white,
            prefixIcon: isEmail
                ? const Icon(Icons.email)
                : (obscureText
                    ? const Icon(Icons.lock)
                    : const Icon(Icons.text_fields)),
          ),
          validator: (value) {
            if (value == null || value.isEmpty) {
              return "กรุณากรอก$label";
            }
            if (isEmail && !RegExp(r'\S+@\S+\.\S+').hasMatch(value)) {
              return "กรุณากรอกอีเมลที่ถูกต้อง";
            }
            return null;
          },
        ),
      ],
    );
  }
}

Future<void> updateUserProfile({
  required String name,
  required String surname,
  required String address,
  required String phone,
}) async {
  try {
    String uid = FirebaseAuth.instance.currentUser!.uid;
    await FirebaseFirestore.instance.collection('Customers').doc(uid).update({
      'name': name,
      'surname': surname,
      'address': address,
      'phone': phone,
      'updatedAt': FieldValue.serverTimestamp(),
    });

    print("ข้อมูลส่วนตัวอัปเดตสำเร็จ");
  } catch (e) {
    print("เกิดข้อผิดพลาด: $e");
  }
}
