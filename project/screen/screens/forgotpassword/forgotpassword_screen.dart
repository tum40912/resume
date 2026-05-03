// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter/services.dart';

// class ForgotPasswordScreen extends StatefulWidget {
//   @override
//   _ForgotPasswordScreenState createState() => _ForgotPasswordScreenState();
// }

// class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
//   final TextEditingController emailController = TextEditingController();
//   final FirebaseAuth _auth = FirebaseAuth.instance;

//   /// ✅ ฟังก์ชันส่งลิงก์รีเซ็ตรหัสผ่านไปทางอีเมล
//   Future<void> sendPasswordResetEmail() async {
//     String email = emailController.text.trim();

//     if (!isValidEmail(email)) {
//       _showErrorDialog("อีเมลไม่ถูกต้อง กรุณากรอกใหม่");
//       return;
//     }

//     try {
//       await _auth.sendPasswordResetEmail(email: email);
//       _showSuccessDialog(
//           "ส่งลิงก์รีเซ็ตรหัสผ่านไปที่อีเมลของคุณแล้ว กรุณาตรวจสอบกล่องจดหมาย");
//     } on FirebaseAuthException catch (e) {
//       if (e.code == 'user-not-found') {
//         _showErrorDialog("ไม่พบบัญชีที่ใช้กับอีเมลนี้");
//       } else if (e.code == 'invalid-email') {
//         _showErrorDialog("รูปแบบอีเมลไม่ถูกต้อง");
//       } else {
//         _showErrorDialog("เกิดข้อผิดพลาด: ${e.message}");
//       }
//     } catch (e) {
//       _showErrorDialog("เกิดข้อผิดพลาด: ${e.toString()}");
//     }
//   }

//   /// ✅ ฟังก์ชันตรวจสอบความถูกต้องของอีเมล
//   bool isValidEmail(String email) {
//     RegExp regex = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');
//     return regex.hasMatch(email);
//   }

//   /// ✅ ฟังก์ชันแสดงกล่องข้อความแจ้งข้อผิดพลาด
//   void _showErrorDialog(String message) {
//     showDialog(
//       context: context,
//       builder: (ctx) => AlertDialog(
//         title: const Text("เกิดข้อผิดพลาด"),
//         content: Text(message),
//         actions: [
//           TextButton(
//             onPressed: () {
//               Navigator.of(ctx).pop();
//             },
//             child: const Text("ปิด"),
//           ),
//         ],
//       ),
//     );
//   }

//   /// ✅ ฟังก์ชันแสดงกล่องข้อความแจ้งเตือนสำเร็จ
//   void _showSuccessDialog(String message) {
//     showDialog(
//       context: context,
//       builder: (ctx) => AlertDialog(
//         title: const Text("สำเร็จ"),
//         content: Text(message),
//         actions: [
//           TextButton(
//             onPressed: () {
//               Navigator.of(ctx).pop();
//               Navigator.of(context).pop(); // ปิดหน้าลืมรหัสผ่าน
//             },
//             child: const Text("ตกลง"),
//           ),
//         ],
//       ),
//     );
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text("ลืมรหัสผ่าน"),
//         backgroundColor: Colors.white,
//         elevation: 0,
//       ),
//       body: Container(
//         width: double.infinity,  // ✅ ขยายให้เต็มหน้าจอ
//         height: double.infinity, // ✅ ขยายให้เต็มหน้าจอ
//         decoration: const BoxDecoration(
//           gradient: LinearGradient(
//             colors: [Color.fromARGB(255, 255, 255, 255), Color.fromARGB(255, 255, 202, 102)],
//             begin: Alignment.topCenter,
//             end: Alignment.bottomCenter,
//           ),
//         ),
//         child: Center( // ✅ จัดวางให้อยู่ตรงกลางของหน้าจอ
//           child: Padding(
//             padding: const EdgeInsets.all(16.0),
//             child: Container(
//               padding: const EdgeInsets.all(16.0),
//               width: 350, // ✅ กำหนดความกว้างของฟอร์ม
//               decoration: BoxDecoration(
//                 color: Colors.white,
//                 borderRadius: BorderRadius.circular(16),
//                 boxShadow: [
//                   BoxShadow(
//                     color: Colors.grey.withOpacity(0.3),
//                     spreadRadius: 4,
//                     blurRadius: 8,
//                     offset: const Offset(0, 2),
//                   ),
//                 ],
//               ),
//               child: Column(
//                 mainAxisSize: MainAxisSize.min,
//                 crossAxisAlignment: CrossAxisAlignment.start,
//                 children: [
//                   const Center(
//                     child: Text(
//                       "รีเซ็ตรหัสผ่าน",
//                       style: TextStyle(
//                         fontSize: 24,
//                         fontWeight: FontWeight.bold,
//                         color: Colors.orange,
//                       ),
//                     ),
//                   ),
//                   const SizedBox(height: 20),
//                   TextField(
//                     controller: emailController,
//                     keyboardType: TextInputType.emailAddress,
//                     inputFormatters: [
//                       FilteringTextInputFormatter.deny(RegExp(r'\s')), // ห้ามเว้นวรรค
//                     ],
//                     decoration: InputDecoration(
//                       labelText: "อีเมล",
//                       border: OutlineInputBorder(
//                         borderRadius: BorderRadius.circular(8),
//                       ),
//                       filled: true,
//                       fillColor: Colors.orange.withOpacity(0.1),
//                     ),
//                   ),
//                   const SizedBox(height: 20),
//                   Center(
//                     child: ElevatedButton(
//                       onPressed: sendPasswordResetEmail,
//                       style: ElevatedButton.styleFrom(
//                         backgroundColor: Colors.orange,
//                         shape: RoundedRectangleBorder(
//                           borderRadius: BorderRadius.circular(12),
//                         ),
//                         padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 24),
//                       ),
//                       child: const Text(
//                         "ส่งลิงก์รีเซ็ตรหัสผ่าน",
//                         style: TextStyle(fontSize: 16, color: Colors.white),
//                       ),
//                     ),
//                   ),
//                 ],
//               ),
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }
