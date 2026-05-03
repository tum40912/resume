import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:krua_pa_ree/screens/base64/base65_image_picker.dart';

class CartScreen extends StatefulWidget {
  @override
  _CartScreenState createState() => _CartScreenState();
}

class _CartScreenState extends State<CartScreen> {
  final user = FirebaseAuth.instance.currentUser;
  final GlobalKey<ScaffoldMessengerState> _scaffoldMessengerKey =
      GlobalKey<ScaffoldMessengerState>();
  bool isSelecting = false; // ✅ โหมดเลือกหลายรายการ
  List<String> selectedItems = []; // ✅ เก็บสินค้าที่ถูกเลือก
  double totalPrice = 0.0; // ตัวแปรเก็บราคารวม

  Future<void> _removeItem(String cartItemId) async {
    await FirebaseFirestore.instance
        .collection('Cart')
        .doc(cartItemId)
        .delete();
  }

  bool isDialogShown = false;

  // ฟังก์ชันสำหรับแสดง Dialog ให้ผู้ใช้เลือกไฟล์สลิป
  // ฟังก์ชันแสดง Dialog ที่รับทั้ง BuildContext และ orderId (String)
  void _showUploadSlipDialog(BuildContext context, String orderId) {
    showDialog(
      context: context,
      barrierDismissible: false, // ❌ บังคับให้ลูกค้าต้องเลือก
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Row(
            children: const [
              Icon(Icons.upload_file, color: Colors.blue, size: 28),
              SizedBox(width: 8),
              Text(
                "อัปโหลดสลิปการโอนเงิน",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "กรุณาอัปโหลดสลิปเพื่อยืนยันการชำระเงินสำหรับออเดอร์ของคุณ",
                style: TextStyle(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: () async {
                  Navigator.pop(dialogContext); // ปิดป๊อปอัปก่อน
                  _uploadPaymentSlip(orderId); // เรียกฟังก์ชันเพื่ออัปโหลดสลิป
                },
                icon: const Icon(Icons.camera_alt, color: Colors.white),
                label: const Text("อัปโหลดสลิป"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30)),
                ),
              ),
              const SizedBox(height: 16),
              TextButton(
                onPressed: () {
                  Navigator.pop(dialogContext); // ปิดป๊อปอัป
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                          "คุณสามารถอัปโหลดสลิปภายหลังได้จากหน้าประวัติออเดอร์"),
                    ),
                  );
                },
                child: const Text(
                  "อัปโหลดภายหลัง",
                  style: TextStyle(
                      color: Colors.grey, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // ฟังก์ชันอัปโหลดสลิป และเปลี่ยนสถานะเป็น "Payment Completed"
  void _uploadPaymentSlip(String orderId) async {
    // เปิดหน้าจอ Base64ImagePicker ให้ผู้ใช้เลือกสลิป
    final result = await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => Base64ImagePicker1()),
    );

    // ตรวจสอบว่าได้ผลลัพธ์จาก Base64ImagePicker หรือไม่
    if (result != null && result is Map<String, dynamic>) {
      final paymentSlipBytes =
          result['image']; // รับข้อมูลจาก Base64ImagePicker
      String base64Image = base64Encode(paymentSlipBytes);

      try {
        // อัปเดตข้อมูลใน Firestore
        await FirebaseFirestore.instance
            .collection('Orders')
            .doc(orderId)
            .update({
          'paymentSlip': base64Image, // เก็บสลิปการชำระเงินในรูปแบบ Base64
          'paymentStatus': 'รอการตรวจสอบ',
          'status': 'Payment Completed', // เปลี่ยนสถานะเป็น "ชำระเงินสำเร็จ"
        });

        // แจ้งเตือนการอัปโหลดสำเร็จ
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text(
                  "อัปโหลดสลิปสำเร็จ! ออเดอร์ของคุณถูกทำเครื่องหมายเป็น 'ชำระเงินสำเร็จ'")),
        );
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("เกิดข้อผิดพลาด: $e")),
        );
      }
    }
  }

  void showPaymentOptions(BuildContext context, String orderId) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext dialogContext) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          child: Container(
            padding: EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black26,
                  blurRadius: 10,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  "🛍️ เลือกวิธีชำระเงิน",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 15),

                // ปุ่มชำระเงินสด
                StreamBuilder<QuerySnapshot>(
                  stream: FirebaseFirestore.instance
                      .collection('Cart')
                      .where('userId',
                          isEqualTo: FirebaseAuth.instance.currentUser!.uid)
                      .snapshots(),
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    }

                    if (snapshot.hasError) {
                      return const Center(
                          child: Text("เกิดข้อผิดพลาดในการดึงข้อมูล"));
                    }

                    final cartItems = snapshot.data!.docs;
                    double totalPrice = 0.0;

                    // คำนวณราคารวมของสินค้าทั้งหมด
                    for (var cartItem in cartItems) {
                      final cartItemData =
                          cartItem.data() as Map<String, dynamic>;
                      final quantity = cartItemData['quantity'] ?? 1;
                      final basePrice =
                          double.tryParse(cartItemData['price'].toString()) ??
                              0.0;

                      // คำนวณราคาเครื่องเคียง
                      final List<dynamic> sides = cartItemData['sides'] ?? [];
                      final double sidesTotal = sides.fold(0.0, (sum, side) {
                        final double sidePrice =
                            double.tryParse(side['price'].toString()) ?? 0.0;
                        return sum + sidePrice;
                      });

                      // คำนวณราคาตัวเลือกพิเศษ
                      final List<dynamic> specials =
                          cartItemData['specials'] ?? [];
                      final double specialsTotal =
                          specials.fold(0.0, (sum, special) {
                        final double specialPrice =
                            double.tryParse(special['price'].toString()) ?? 0.0;
                        return sum + specialPrice;
                      });

                      // คำนวณราคารวมของสินค้าทั้งหมด
                      double itemTotalPrice =
                          (basePrice + sidesTotal + specialsTotal) * quantity;
                      totalPrice += itemTotalPrice;
                    }

                    return Column(
                      children: [
                        ElevatedButton(
                          onPressed: () async {
                            if (cartItems.isEmpty) {
                              if (mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text("ไม่มีสินค้าในตะกร้า"),
                                  ),
                                );
                              }
                              return;
                            }

                            try {
                              final userId =
                                  FirebaseAuth.instance.currentUser!.uid;
                              final String orderId = await _generateOrderId();

                              final orderRef = FirebaseFirestore.instance
                                  .collection('Orders')
                                  .doc(orderId);

                              await orderRef.set({
                                'orderId': orderId,
                                'userId': userId,
                                'items': cartItems
                                    .map((item) => item.data())
                                    .toList(),
                                'totalPrice': totalPrice,
                                'status': 'Waiting',
                                'paymentMethod':
                                    'cash', // Add this line for payment method
                                'timestamp': FieldValue.serverTimestamp(),
                              });

                              for (var cartItem in cartItems) {
                                await FirebaseFirestore.instance
                                    .collection('Cart')
                                    .doc(cartItem.id)
                                    .delete();
                              }

                              if (mounted) {
                                ScaffoldMessenger.of(context)
                                    .hideCurrentSnackBar();
                              }

                              if (mounted) {
                                showDialog(
                                  context: context,
                                  barrierDismissible: false,
                                  builder: (BuildContext dialogContext) {
                                    return AlertDialog(
                                      title: const Text("สั่งซื้อสำเร็จ! 🎉"),
                                      content: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          const Icon(Icons.check_circle,
                                              color: Colors.green, size: 60),
                                          const SizedBox(height: 10),
                                          const Text("หมายเลขคำสั่งซื้อของคุณ:",
                                              style: TextStyle(fontSize: 16)),
                                          const SizedBox(height: 5),
                                          Text(
                                            orderId,
                                            style: const TextStyle(
                                                fontSize: 20,
                                                fontWeight: FontWeight.bold,
                                                color: Colors.blue),
                                          ),
                                          const SizedBox(height: 10),
                                          const Text(
                                              "ขอบคุณที่ใช้บริการของเรา 😊",
                                              textAlign: TextAlign.center),
                                        ],
                                      ),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.pop(dialogContext);
                                            Future.delayed(
                                                const Duration(
                                                    milliseconds: 300), () {
                                              if (mounted) {
                                                Navigator.pop(context);
                                              }
                                            });
                                          },
                                          child: const Text("ตกลง",
                                              style: TextStyle(fontSize: 16)),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              }
                            } catch (e) {
                              if (mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                        content: Text("เกิดข้อผิดพลาด: $e")));
                              }
                            }
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green,
                            minimumSize: Size(double.infinity, 50),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.money, color: Colors.white, size: 24),
                              SizedBox(width: 10),
                              Text("ชำระเงินสด",
                                  style: TextStyle(
                                      fontSize: 18, color: Colors.white)),
                            ],
                          ),
                        ),

                        // ปุ่มใหม่ "สแกนจ่ายข้าง"
                        SizedBox(height: 10),
                        ElevatedButton(
                          onPressed: () async {
                            // Check if there are items in the cart
                            if (cartItems.isEmpty) {
                              if (mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text("ไม่มีสินค้าในตะกร้า"),
                                  ),
                                );
                              }
                              return;
                            }

                            try {
                              // Get the current user's ID
                              final userId =
                                  FirebaseAuth.instance.currentUser!.uid;
                              // Generate a unique order ID
                              final String orderId = await _generateOrderId();

                              // Reference to the new order in Firestore
                              final orderRef = FirebaseFirestore.instance
                                  .collection('Orders')
                                  .doc(orderId);

                              // Save the order details, including payment method as 'QR' in this case
                              await orderRef.set({
                                'orderId': orderId,
                                'userId': userId,
                                'items': cartItems
                                    .map((item) => item.data())
                                    .toList(),
                                'totalPrice': totalPrice,
                                'status': 'Waiting', // Initial status
                                'paymentMethod':
                                    'QR', // Set the payment method to QR
                                'timestamp': FieldValue.serverTimestamp(),
                              });

                              // Optionally, trigger QR payment screen here
                              showQrPayment(context, orderId);

                              // After processing, remove the items from the cart in Firestore
                              for (var cartItem in cartItems) {
                                await FirebaseFirestore.instance
                                    .collection('Cart')
                                    .doc(cartItem.id)
                                    .delete();
                              }

                              if (mounted) {
                                ScaffoldMessenger.of(context)
                                    .hideCurrentSnackBar();
                              }

                              // Show success dialog
                            } catch (e) {
                              // Handle any errors that occur during order creation
                              if (mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                        content: Text("เกิดข้อผิดพลาด: $e")));
                              }
                            }
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blueAccent,
                            minimumSize: Size(double.infinity, 50),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.qr_code_2,
                                  color: Colors.white, size: 24),
                              SizedBox(width: 10),
                              Text("สแกนจ่าย",
                                  style: TextStyle(
                                      fontSize: 18, color: Colors.white)),
                            ],
                          ),
                        ),
                      ],
                    );
                  },
                )
              ],
            ),
          ),
        );
      },
    );
  }

  // ✅ แสดง QR Code จากไฟล์ใน assets และเพิ่มปุ่ม "เสร็จสิ้น"
  void showQrPayment(BuildContext context, String orderId) {
    showDialog(
      context: context,
      barrierDismissible: false, // ป้องกันการปิดโดยกดข้างนอก
      builder: (BuildContext dialogContext) {
        return Dialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          child: Container(
            padding: EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black26,
                  blurRadius: 10,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  "📷 สแกนเพื่อชำระเงิน",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 15),

                // ✅ แสดง QR Code
                Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(15),
                    border: Border.all(color: Colors.grey.shade300, width: 2),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(15),
                    child: Image.asset(
                      'assets/images/QR.png', // ✅ ใช้ภาพ QR จาก assets
                      height: 200,
                      width: 200,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),

                SizedBox(height: 15),

                // ✅ แสดงข้อความแจ้งเตือน
                Text(
                  "กรุณาสแกนจ่าย ${totalPrice.toInt()} บาท",
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87),
                  textAlign: TextAlign.center,
                ),

                SizedBox(height: 20),

                // ✅ ปุ่มกด
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // ปุ่มปิด
                    ElevatedButton.icon(
                      onPressed: () {
                        // ปิด Popup ที่เปิดอยู่
                        Navigator.pop(dialogContext);

                        // รีเฟรชหน้า CartScreen
                        Navigator.pushReplacement(
                          context,
                          MaterialPageRoute(
                              builder: (context) =>
                                  CartScreen()), // ไปยังหน้า CartScreen และรีเฟรช
                        );
                      },
                      icon: Icon(Icons.close, color: Colors.white),
                      label: Text("ปิด"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10)),
                        padding:
                            EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      ),
                    ),

                    // ปุ่มเสร็จสิ้น
                    ElevatedButton.icon(
                      onPressed: () async {
                        // เรียกฟังก์ชันแสดง Dialog สำหรับอัพโหลด Payment Slip
                        _showUploadSlipDialog(context, orderId);
                      },
                      icon: Icon(Icons.check_circle, color: Colors.white),
                      label: Text("เสร็จสิ้น"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        padding:
                            EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      ),
                    )
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  void updateOrderStatus(
      BuildContext context, String paymentMethod, String orderId) async {
    try {
      final userId = FirebaseAuth.instance.currentUser!.uid;

      // สร้าง order ใน Firebase
      final orderRef =
          FirebaseFirestore.instance.collection('Orders').doc(orderId);
      await orderRef.update({
        'paymentMethod': paymentMethod,
        'status': 'Completed',
        'timestamp': FieldValue.serverTimestamp(),
      });

      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("การชำระเงินสำเร็จ")));
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("เกิดข้อผิดพลาด: $e")));
    }
  }

  Future<void> _showConfirmDeleteDialog() async {
    if (isDialogShown) return; // ไม่ให้เปิด Dialog ซ้ำถ้ามีอยู่แล้ว
    isDialogShown = true; // ตั้งค่าให้แสดงแล้ว

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text(
            "ยืนยันการลบ",
            style: TextStyle(
              color: Colors.blueAccent,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          content: const Text(
            "คุณแน่ใจหรือไม่ว่าต้องการลบรายการที่เลือกทั้งหมด?",
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w400,
              color: Colors.black,
            ),
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          backgroundColor: Colors.white,
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                isDialogShown = false; // รีเซ็ต flag เมื่อปิด Dialog
              },
              child: const Text(
                "ยกเลิก",
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.blueAccent,
                ),
              ),
            ),
            TextButton(
              onPressed: () async {
                await _removeSelectedItems();
                Navigator.pop(context);
                isDialogShown = false; // รีเซ็ต flag หลังจากลบรายการ
              },
              child: const Text(
                "ยืนยัน",
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.red,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Future<void> _removeSelectedItems() async {
    if (selectedItems.isEmpty) return;

    final batch = FirebaseFirestore.instance.batch();

    // ลบรายการที่เลือกทั้งหมดจาก Firebase
    for (String cartItemId in selectedItems) {
      batch.delete(
        FirebaseFirestore.instance.collection('Cart').doc(cartItemId),
      );
    }

    try {
      await batch
          .commit(); // ใช้ batch.commit() เพื่อดำเนินการทั้งหมดในครั้งเดียว
      setState(() {
        selectedItems.clear(); // เคลียร์รายการที่เลือก
        isSelecting = false; // ปิดโหมดเลือกหลายรายการ
      });

      // แสดง Dialog สำเร็จหลังจากการลบเสร็จสิ้น
      if (!isDialogShown) {
        // ป้องกันการแสดง Dialog ซ้ำ
        isDialogShown = true;
        showDialog(
          context: context,
          barrierDismissible: false, // ปิดการกดข้างนอกเพื่อปิด Dialog
          builder: (BuildContext dialogContext) {
            return AlertDialog(
              title: const Text(
                "สำเร็จ!",
                style: TextStyle(
                  color: Colors.blueAccent,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              content: const Text(
                "ลบรายการที่เลือกเรียบร้อยแล้ว",
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w400,
                  color: Colors.black,
                ),
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15), // ขอบโค้งมน
              ),
              backgroundColor: Colors.white, // สีพื้นหลังของ Dialog
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(dialogContext); // ปิด Dialog
                    isDialogShown = false; // รีเซ็ต flag
                  },
                  child: const Text(
                    "ตกลง",
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.blueAccent, // เปลี่ยนสีของปุ่ม
                    ),
                  ),
                ),
              ],
            );
          },
        );
      }
    } catch (e) {
      // หากเกิดข้อผิดพลาด
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("เกิดข้อผิดพลาดในการลบ: $e")),
      );
    }
  }

  // ✅ เพิ่ม/ลดจำนวนสินค้า
  Future<void> _updateQuantity(String cartItemId, int newQuantity) async {
    if (newQuantity <= 0) {
      await _removeItem(cartItemId);
    } else {
      await FirebaseFirestore.instance
          .collection('Cart')
          .doc(cartItemId)
          .update({'quantity': newQuantity});
    }
  }

  // ฟังก์ชันสร้าง Order ID ในรูปแบบ Order000001, Order000002, ...
  Future<String> _generateOrderId() async {
    final counterRef =
        FirebaseFirestore.instance.collection('Orders').doc('order_count');

    return FirebaseFirestore.instance.runTransaction((transaction) async {
      final snapshot = await transaction.get(counterRef);

      int currentOrderNumber = 1;

      if (snapshot.exists && snapshot.data() != null) {
        currentOrderNumber = (snapshot.data()!['count'] ?? 0) + 1;
      }

      // อัปเดตค่า count ใน Firestore
      transaction.set(counterRef, {'count': currentOrderNumber});

      // สร้าง Order ID
      return "Order${currentOrderNumber.toString().padLeft(6, '0')}";
    });
  }

  @override
  Widget build(BuildContext context) {
    if (user == null) {
      return Scaffold(
        appBar: AppBar(title: const Text("ตะกร้าของคุณ")),
        body: const Center(child: Text("กรุณาเข้าสู่ระบบก่อน")),
      );
    }

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
              "ตะกร้าของคุณ",
              style: TextStyle(
                fontFamily: "assets/fonts/ChakraPetch-Bold.ttf",
                color: Color.fromARGB(255, 0, 0, 0),
                fontWeight: FontWeight.bold,
              ),
            ),
            centerTitle: true, // จัดกึ่งกลางข้อความ
            elevation: 5, // เพิ่มเงา
            actions: [
              IconButton(
                icon: Icon(isSelecting
                    ? Icons.close
                    : Icons.edit), // 📝 ปุ่มเปิดโหมดเลือก
                onPressed: () {
                  setState(() {
                    isSelecting = !isSelecting;
                    if (!isSelecting)
                      selectedItems
                          .clear(); // ถ้าออกจากโหมดเลือก ให้เคลียร์รายการที่เลือก
                  });
                },
              ),
            ],
          ),
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
        child: StreamBuilder<QuerySnapshot>(
          stream: FirebaseFirestore.instance
              .collection('Cart')
              .where('userId', isEqualTo: user!.uid)
              .snapshots(),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.data!.docs.isEmpty) {
              return const Center(child: Text("ไม่มีสินค้าในตะกร้า"));
            }

            final cartItems = snapshot.data!.docs;
            double newTotalPrice = 0.0; // ตัวแปรเก็บราคาชั่วคราว

            // คำนวณราคารวมของสินค้าทั้งหมด
            for (var cartItem in cartItems) {
              final cartItemData = cartItem.data() as Map<String, dynamic>;
              final quantity = cartItemData['quantity'] ?? 1;
              final basePrice =
                  double.tryParse(cartItemData['price'].toString()) ?? 0.0;

              // คำนวณราคาเครื่องเคียง
              final List<dynamic> sides = cartItemData['sides'] ?? [];
              final double sidesTotal = sides.fold(0.0, (sum, side) {
                final double sidePrice =
                    double.tryParse(side['price'].toString()) ?? 0.0;
                return sum + sidePrice;
              });

              // คำนวณราคาตัวเลือกพิเศษ
              final List<dynamic> specials = cartItemData['specials'] ?? [];
              final double specialsTotal = specials.fold(0.0, (sum, special) {
                final double specialPrice =
                    double.tryParse(special['price'].toString()) ?? 0.0;
                return sum + specialPrice;
              });

              // คำนวณราคารวมของสินค้าทั้งหมด
              double itemTotalPrice =
                  (basePrice + sidesTotal + specialsTotal) * quantity;
              newTotalPrice += itemTotalPrice; // รวมราคารวมทั้งหมด
            }

            // ใช้ Future.microtask เพื่อให้ setState() ทำงานหลังจากที่ build เสร็จ
            Future.microtask(() {
              setState(() {
                totalPrice = newTotalPrice;
              });
            });

            return Column(
              children: [
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16.0),
                    itemCount: cartItems.length,
                    itemBuilder: (context, index) {
                      final cartItem =
                          cartItems[index].data() as Map<String, dynamic>;
                      final cartItemId = cartItems[index].id;
                      final name = cartItem['name']?.toString() ?? 'ไม่มีชื่อ';
                      final quantity = cartItem['quantity'] ?? 1;

                      // คำนวณราคาเมนูหลัก (base price)
                      final double basePrice =
                          double.tryParse(cartItem['price'].toString()) ?? 0.0;

                      // ตรวจสอบเครื่องเคียงและตัวเลือกพิเศษ
                      final List<dynamic> sides =
                          (cartItem['sides'] as List<dynamic>?) ?? [];
                      final List<dynamic> specials =
                          (cartItem['specials'] as List<dynamic>?) ?? [];
                      final String note = cartItem['note']?.toString() ?? '';

                      // คำนวณราคาเครื่องเคียง
                      double sidesTotal = sides.fold(0.0, (sum, side) {
                        final double sidePrice =
                            double.tryParse(side['price'].toString()) ?? 0.0;
                        return sum + sidePrice;
                      });

                      // คำนวณราคาตัวเลือกพิเศษ
                      double specialsTotal = specials.fold(0.0, (sum, special) {
                        final double specialPrice =
                            double.tryParse(special['price'].toString()) ?? 0.0;
                        return sum + specialPrice;
                      });

                      return Card(
                        elevation: 3,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(12.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                name,
                                style: const TextStyle(
                                    fontSize: 18, fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 5),
                              Text(
                                  "จำนวน: $quantity | ราคา: ${basePrice.toInt()} บาท"),
                              const SizedBox(height: 5),
                              if (sides.isNotEmpty) ...[
                                const Text("เพิ่มเติม:",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold)),
                                ...sides.map((side) {
                                  return Text(
                                      "- ${side['name']} (+${side['price']} บาท)");
                                }).toList(),
                              ],
                              if (specials.isNotEmpty) ...[
                                const SizedBox(height: 5),
                                const Text("ตัวเลือกพิเศษ:",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold)),
                                ...specials.map((special) {
                                  return Text(
                                      "- ${special['name']} (+${special['price']} บาท)");
                                }).toList(),
                              ],
                              if (note.isNotEmpty) ...[
                                const SizedBox(height: 5),
                                Text("หมายเหตุ: $note",
                                    style: const TextStyle(
                                        fontStyle: FontStyle.italic)),
                              ],
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Row(
                                    children: [
                                      IconButton(
                                        onPressed: () => _updateQuantity(
                                            cartItemId, quantity - 1),
                                        icon: const Icon(Icons.remove_circle,
                                            color: Colors.red),
                                      ),
                                      Text(quantity.toString(),
                                          style: const TextStyle(fontSize: 18)),
                                      IconButton(
                                        onPressed: () => _updateQuantity(
                                            cartItemId, quantity + 1),
                                        icon: const Icon(Icons.add_circle,
                                            color: Colors.green),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                              Row(
                                children: [
                                  if (isSelecting)
                                    Checkbox(
                                      value: selectedItems.contains(cartItemId),
                                      onChanged: (selected) {
                                        setState(() {
                                          if (selected == true) {
                                            selectedItems.add(cartItemId);
                                          } else {
                                            selectedItems.remove(cartItemId);
                                          }
                                        });
                                      },
                                    ),
                                ],
                              ),
                              Align(
                                alignment: Alignment.centerRight,
                                child: IconButton(
                                  icon: const Icon(Icons.delete,
                                      color: Colors.red),
                                  onPressed: () {
                                    _removeItem(cartItemId);
                                  },
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
                Builder(
                  builder: (context) {
                    return Container(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            "ราคารวม: ${totalPrice.toInt()} บาท", // ใช้ toInt() เพื่อแสดงจำนวนเต็ม
                            style: const TextStyle(
                                fontSize: 18, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 10),
                          // ElevatedButton(
                          //   onPressed: () async {
                          //     if (cartItems.isEmpty) {
                          //       if (mounted) {
                          //         ScaffoldMessenger.of(context).showSnackBar(
                          //           const SnackBar(
                          //               content: Text("ไม่มีสินค้าในตะกร้า")),
                          //         );
                          //       }
                          //       return;
                          //     }

                          //     try {
                          //       final userId =
                          //           FirebaseAuth.instance.currentUser!.uid;
                          //       final String orderId = await _generateOrderId();

                          //       final orderRef = FirebaseFirestore.instance
                          //           .collection('Orders')
                          //           .doc(orderId);

                          //       await orderRef.set({
                          //         'orderId': orderId,
                          //         'userId': userId,
                          //         'items': cartItems
                          //             .map((item) => item.data())
                          //             .toList(),
                          //         'totalPrice': totalPrice,
                          //         'status': 'Waiting',
                          //         'timestamp': FieldValue.serverTimestamp(),
                          //       });

                          //       for (var cartItem in cartItems) {
                          //         await FirebaseFirestore.instance
                          //             .collection('Cart')
                          //             .doc(cartItem.id)
                          //             .delete();
                          //       }

                          //       if (mounted) {
                          //         ScaffoldMessenger.of(context)
                          //             .hideCurrentSnackBar();
                          //       }

                          //       if (mounted) {
                          //         showDialog(
                          //           context: context,
                          //           barrierDismissible: false,
                          //           builder: (BuildContext dialogContext) {
                          //             return AlertDialog(
                          //               title: const Text("สั่งซื้อสำเร็จ! 🎉"),
                          //               content: Column(
                          //                 mainAxisSize: MainAxisSize.min,
                          //                 children: [
                          //                   const Icon(Icons.check_circle,
                          //                       color: Colors.green, size: 60),
                          //                   const SizedBox(height: 10),
                          //                   const Text(
                          //                       "หมายเลขคำสั่งซื้อของคุณ:",
                          //                       style: TextStyle(fontSize: 16)),
                          //                   const SizedBox(height: 5),
                          //                   Text(orderId,
                          //                       style: const TextStyle(
                          //                           fontSize: 20,
                          //                           fontWeight: FontWeight.bold,
                          //                           color: Colors.blue)),
                          //                   const SizedBox(height: 10),
                          //                   const Text(
                          //                       "ขอบคุณที่ใช้บริการของเรา 😊",
                          //                       textAlign: TextAlign.center),
                          //                 ],
                          //               ),
                          //               actions: [
                          //                 TextButton(
                          //                   onPressed: () {
                          //                     Navigator.pop(dialogContext);
                          //                     Future.delayed(
                          //                         const Duration(
                          //                             milliseconds: 300), () {
                          //                       if (mounted) {
                          //                         Navigator.pop(context);
                          //                       }
                          //                     });
                          //                   },
                          //                   child: const Text("ตกลง",
                          //                       style: TextStyle(fontSize: 16)),
                          //                 ),
                          //               ],
                          //             );
                          //           },
                          //         );
                          //       }
                          //     } catch (e) {
                          //       if (mounted) {
                          //         ScaffoldMessenger.of(context).showSnackBar(
                          //           SnackBar(
                          //               content: Text("เกิดข้อผิดพลาด: $e")),
                          //         );
                          //       }
                          //     }
                          //   },
                          //   style: ElevatedButton.styleFrom(
                          //     backgroundColor: Colors.green,
                          //     padding: const EdgeInsets.symmetric(
                          //         vertical: 12, horizontal: 50),
                          //     shape: RoundedRectangleBorder(
                          //         borderRadius: BorderRadius.circular(20)),
                          //   ),
                          //   child: const Text("สั่งซื้อ",
                          //       style: TextStyle(
                          //           fontSize: 18, color: Colors.white)),
                          // ),
                          ElevatedButton(
                            onPressed: () async {
                              if (cartItems.isEmpty) {
                                if (mounted) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                        content: Text("ไม่มีสินค้าในตะกร้า")),
                                  );
                                }
                                return;
                              }

                              try {
                                final userId =
                                    FirebaseAuth.instance.currentUser!.uid;
                                final String orderId =
                                    await _generateOrderId(); // สร้าง orderId

                                // แสดงตัวเลือกการชำระเงิน
                                showPaymentOptions(context, orderId);
                              } catch (e) {
                                if (mounted) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                        content: Text("เกิดข้อผิดพลาด: $e")),
                                  );
                                }
                              }
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.green,
                              padding: const EdgeInsets.symmetric(
                                  vertical: 12, horizontal: 50),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20),
                              ),
                            ),
                            child: const Text(
                              "สั่งซื้อ",
                              style:
                                  TextStyle(fontSize: 18, color: Colors.white),
                            ),
                          ),
                          if (isSelecting && selectedItems.isNotEmpty)
                            Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: ElevatedButton(
                                onPressed:
                                    _showConfirmDeleteDialog, // เรียกฟังก์ชันยืนยันการลบ
                                style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.red),
                                child: const Text("ลบที่เลือก",
                                    style: TextStyle(
                                        fontSize: 18, color: Colors.white)),
                              ),
                            )
                        ],
                      ),
                    );
                  },
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
