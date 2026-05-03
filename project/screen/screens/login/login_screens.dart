import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:krua_pa_ree/screens/homeowner/homeowner_screens.dart';
import 'package:krua_pa_ree/screens/profile/profile_screen.dart';
import 'package:krua_pa_ree/screens/register/register_screen.dart';

import '../home/home_screens.dart';
import '../homerider/homerider_screen.dart';

class LoginScreen extends StatelessWidget {
  LoginScreen({Key? key}) : super(key: key);

  final _formKeyLogin = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  final String ownerEmail = "owner@gmail.com";
  Future<void> _login(BuildContext context) async {
    final String email = _emailController.text.trim();
    final String password = _passwordController.text.trim();

    if (email.isEmpty || password.isEmpty) {
      _showErrorDialog(context, "กรุณากรอกข้อมูลให้ครบถ้วน");
      return;
    }

    try {
      UserCredential userCredential =
          await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      if (userCredential.user != null) {
        if (userCredential.user!.email == ownerEmail) {
          await FirebaseFirestore.instance
              .collection('Owners')
              .doc(userCredential.user!.uid)
              .set({
            'email': email,
            'role': 'Owner',
            'createdAt': Timestamp.now(),
          });

          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => HomeOwnerScreen()),
          );
        } else {
          DocumentSnapshot userDoc = await FirebaseFirestore.instance
              .collection('Customers')
              .doc(userCredential.user!.uid)
              .get();

          if (userDoc.exists) {
            if (userDoc.data() != null &&
                (userDoc.data() as Map<String, dynamic>).containsKey('name')) {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (context) => HomeScreen()),
              );
            } else {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (context) => ProfileScreen()),
              );
            }
          } else {
            DocumentSnapshot employeeDoc = await FirebaseFirestore.instance
                .collection('Employees')
                .doc(userCredential.user!.uid)
                .get();

            if (employeeDoc.exists) {
              String role =
                  (employeeDoc.data() as Map<String, dynamic>)['role'] ?? '';

              if (role == 'rider') {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => HomeRiderScreen()),
                );
                return;
              }
            }

            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => ProfileScreen()),
            );
          }
        }
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        _showErrorDialog(context, "ไม่พบผู้ใช้งาน กรุณาตรวจสอบอีเมลอีกครั้ง");
      } else if (e.code == 'wrong-password') {
        _showErrorDialog(context, "รหัสผ่านไม่ถูกต้อง");
      } else {
        _showErrorDialog(context, "กรุณากรอกอีเมลหรือรหัสผ่านให้ถูกต้อง");
      }
    } catch (e) {
      _showErrorDialog(context, "เกิดข้อผิดพลาด: ${e.toString()}");
    }
  }

  Future<void> _saveOwnerData(String userId) async {
    try {
      final ownerRef =
          FirebaseFirestore.instance.collection('Owners').doc(userId);
      await ownerRef.set({
        'role': 'Owner',
        'email': ownerEmail,
        'uid': userId,
      });
    } catch (e) {
      print("Error saving owner data: $e");
    }
  }

  void _showForgotPasswordDialog(BuildContext context) {
    final TextEditingController emailController = TextEditingController();

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("ลืมรหัสผ่าน"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text("กรุณากรอกอีเมลของคุณเพื่อรับลิงก์ตั้งค่ารหัสผ่านใหม่"),
            const SizedBox(height: 10),
            TextField(
              controller: emailController,
              decoration: const InputDecoration(
                hintText: "อีเมลของคุณ",
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(ctx).pop();
            },
            child: const Text("ยกเลิก"),
          ),
          ElevatedButton(
            onPressed: () async {
              final email = emailController.text.trim();
              if (email.isEmpty ||
                  !RegExp(r"^[^@\s]+@[^@\s]+\.[^@\s]+").hasMatch(email)) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("กรุณากรอกอีเมลที่ถูกต้อง")),
                );
                return;
              }

              try {
                await FirebaseAuth.instance
                    .sendPasswordResetEmail(email: email);
                Navigator.of(ctx).pop();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                      content:
                          Text("ส่งลิงก์รีเซ็ตรหัสผ่านไปยังอีเมลของคุณแล้ว")),
                );
              } catch (e) {
                Navigator.of(ctx).pop();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text("เกิดข้อผิดพลาด: ${e.toString()}")),
                );
              }
            },
            child: const Text("ส่งลิงก์รีเซ็ตรหัสผ่าน"),
          ),
        ],
      ),
    );
  }

  void _showErrorDialog(BuildContext context, String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("เกิดข้อผิดพลาด"),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(ctx).pop();
            },
            child: const Text("ปิด"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Color.fromARGB(255, 252, 220, 179)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: GestureDetector(
          onTap: () {
            FocusScope.of(context).unfocus();
          },
          child: Center(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(10.0),
                child: Card(
                  elevation: 12,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                  color: const Color(0x7CFFFFFF),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        vertical: 40, horizontal: 20),
                    child: Column(
                      children: [
                        Image.asset(
                          'assets/images/krua pa ree.png',
                          height: 150,
                          fit: BoxFit.contain,
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          "เข้าสู่ระบบ",
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.black87,
                          ),
                        ),
                        const SizedBox(height: 30),
                        Form(
                          key: _formKeyLogin,
                          child: Column(
                            children: [
                              TextFormField(
                                controller: _emailController,
                                decoration: const InputDecoration(
                                  prefixIcon: Icon(Icons.email_outlined),
                                  hintText: "Email",
                                  border: OutlineInputBorder(),
                                ),
                                keyboardType: TextInputType.emailAddress,
                                validator: (value) {
                                  if (value == null || value.isEmpty) {
                                    return "กรุณากรอกอีเมล์";
                                  } else if (!RegExp(
                                          r"^[^@\s]+@[^@\s]+\.[^@\s]+")
                                      .hasMatch(value)) {
                                    return "กรุณากรอกอีเมลให้ถูกต้อง";
                                  }
                                  return null;
                                },
                              ),
                              const SizedBox(height: 10),
                              TextFormField(
                                controller: _passwordController,
                                obscureText: true,
                                decoration: const InputDecoration(
                                  prefixIcon: Icon(Icons.lock_outline),
                                  hintText: "Password",
                                  border: OutlineInputBorder(),
                                ),
                                validator: (value) {
                                  if (value == null || value.isEmpty) {
                                    return "กรุณากรอกรหัสผ่าน";
                                  } else if (value.length < 6) {
                                    return "รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร";
                                  }
                                  return null;
                                },
                              ),
                              const SizedBox(height: 10),
                              Align(
                                alignment: Alignment.centerRight,
                                child: TextButton(
                                  onPressed: () {
                                    _showForgotPasswordDialog(context);
                                  },
                                  child: const Text(
                                    "ลืมรหัสผ่าน?",
                                    style: TextStyle(
                                        fontSize: 16, color: Colors.orange),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 10),
                              ElevatedButton(
                                onPressed: () {
                                  if (_formKeyLogin.currentState!.validate()) {
                                    _login(context);
                                  }
                                },
                                style: ElevatedButton.styleFrom(
                                  minimumSize: const Size.fromHeight(50),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                ),
                                child: const Text(
                                  "เข้าสู่ระบบ",
                                  style: TextStyle(
                                    color: Colors.black,
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                              const SizedBox(height: 20),
                              Row(
                                children: const [
                                  Expanded(
                                    child: Divider(
                                      thickness: 1,
                                      color: Colors.orange,
                                    ),
                                  ),
                                  Padding(
                                    padding:
                                        EdgeInsets.symmetric(horizontal: 8.0),
                                    child: Text("หรือเข้าสู่ระบบด้วย"),
                                  ),
                                  Expanded(
                                    child: Divider(
                                      thickness: 1,
                                      color: Colors.orange,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              const SizedBox(height: 20),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Text("ยังไม่มีบัญชีใช่ไหม? "),
                                  GestureDetector(
                                    onTap: () {
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                            builder: (context) =>
                                                RegisterScreen()),
                                      );
                                    },
                                    child: const Text(
                                      "สมัครฟรี",
                                      style: TextStyle(
                                        color: Colors.orange,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
