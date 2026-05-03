import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:krua_pa_ree/screens/base64/base64_image_picker.dart';

class OrderUseScreen extends StatefulWidget {
  @override
  _OrderUseScreenState createState() => _OrderUseScreenState();
}

class _OrderUseScreenState extends State<OrderUseScreen> {
  final user = FirebaseAuth.instance.currentUser;

  // ฟังก์ชันแปลงสถานะเป็นภาษาไทย
  // ฟังก์ชันแปลงสถานะเป็นภาษาไทย
  String translateStatus(String status) {
    switch (status) {
      case 'Waiting':
        return 'รอรับออเดอร์';
      case 'Accepted':
        return 'กำลังปรุง';
      case 'Completed':
        return 'ปรุงเสร็จสิ้น';
      case 'Delivered':
        return 'จัดส่งแล้ว';
      case 'Payment Completed':
        return 'ชำระเงินสำเร็จ';
      default:
        return 'ไม่ทราบสถานะ';
    }
  }

  // 📌 ฟังก์ชันแสดงป๊อปอัปอัปโหลดสลิป
  // void _showUploadSlipDialog(String orderId) {
  //   showDialog(
  //     context: context,
  //     barrierDismissible: false, // ❌ บังคับให้ลูกค้าต้องเลือก
  //     builder: (BuildContext dialogContext) {
  //       return AlertDialog(
  //         shape:
  //             RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
  //         title: Row(
  //           children: const [
  //             Icon(Icons.upload_file, color: Colors.blue, size: 28),
  //             SizedBox(width: 8),
  //             Text("อัปโหลดสลิปการโอนเงิน",
  //                 style: TextStyle(fontWeight: FontWeight.bold)),
  //           ],
  //         ),
  //         content: Column(
  //           mainAxisSize: MainAxisSize.min,
  //           children: [
  //             const Text(
  //               "กรุณาอัปโหลดสลิปเพื่อยืนยันการชำระเงินสำหรับออเดอร์ของคุณ",
  //               style: TextStyle(fontSize: 16),
  //               textAlign: TextAlign.center,
  //             ),
  //             const SizedBox(height: 16),
  //             ElevatedButton.icon(
  //               onPressed: () async {
  //                 Navigator.pop(dialogContext); // ปิดป๊อปอัปก่อน
  //                 _uploadPaymentSlip(orderId);
  //               },
  //               icon: const Icon(Icons.camera_alt, color: Colors.white),
  //               label: const Text("อัปโหลดสลิป"),
  //               style: ElevatedButton.styleFrom(
  //                 backgroundColor: Colors.green,
  //                 shape: RoundedRectangleBorder(
  //                     borderRadius: BorderRadius.circular(30)),
  //               ),
  //             ),
  //             const SizedBox(height: 16),
  //             TextButton(
  //               onPressed: () {
  //                 Navigator.pop(dialogContext); // ปิดป๊อปอัป
  //                 ScaffoldMessenger.of(context).showSnackBar(
  //                   SnackBar(
  //                       content: Text(
  //                           "คุณสามารถอัปโหลดสลิปภายหลังได้จากหน้าประวัติออเดอร์")),
  //                 );
  //               },
  //               child: const Text("อัปโหลดภายหลัง",
  //                   style: TextStyle(
  //                       color: Colors.grey, fontWeight: FontWeight.bold)),
  //             ),
  //           ],
  //         ),
  //       );
  //     },
  //   );
  // }

  // 📌 ฟังก์ชันอัปโหลดสลิป และเปลี่ยนสถานะเป็น "Payment Completed"
  // void _uploadPaymentSlip(String orderId) async {
  //   final result = await Navigator.push(
  //     context,
  //     MaterialPageRoute(builder: (context) => Base64ImagePicker()),
  //   );

  //   if (result != null && result is Uint8List) {
  //     String base64Image = base64Encode(result);

  //     await FirebaseFirestore.instance
  //         .collection('Orders')
  //         .doc(orderId)
  //         .update({
  //       'paymentSlip': base64Image,
  //       'paymentStatus': 'รอการตรวจสอบ',
  //       'status': 'Payment Completed', // ✅ เปลี่ยนสถานะเป็น "ชำระเงินสำเร็จ"
  //     });

  //     // ✅ แจ้งเตือน และลบออกจากหน้า UI
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(
  //           content: Text(
  //               "อัปโหลดสลิปสำเร็จ! ออเดอร์ของคุณถูกทำเครื่องหมายเป็น 'ชำระเงินสำเร็จ'")),
  //     );
  //   }
  // }

  // ฟังก์ชันยกเลิกออเดอร์ใน Firestore
  void showCancelOrderDialog(BuildContext context, String orderId) {
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
              Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
              SizedBox(width: 8),
              Text(
                "ยืนยันการยกเลิก",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "คุณต้องการยกเลิกออเดอร์นี้หรือไม่?",
                style: TextStyle(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                "Order ID: $orderId",
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
                textAlign: TextAlign.center,
              ),
            ],
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
                  // ยกเลิกออเดอร์ใน Firestore
                  await FirebaseFirestore.instance
                      .collection('Orders')
                      .doc(orderId)
                      .update({'status': 'Cancelled'});

                  Navigator.pop(dialogContext); // ปิดป็อปอัป

                  // แสดงข้อความยืนยัน
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text("Order $orderId ถูกยกเลิกเรียบร้อย"),
                      backgroundColor: Colors.green,
                      behavior: SnackBarBehavior.floating,
                    ),
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

  // ฟังก์ชันคำนวณยอดรวม
  double calculateTotal(List items) {
    if (items == null || items.isEmpty) return 0.0;
    double total = 0.0;
    for (var item in items) {
      final price = double.tryParse(item['price'].toString()) ?? 0.0;
      final quantity = item['quantity'] ?? 1;
      total += price * quantity;
    }
    return total;
  }

  // ฟังก์ชันจัดการกับ totalPrice
  double _getTotalPrice(Map<String, dynamic> order) {
    if (order['totalPrice'] == null) {
      return 0.0; // ถ้าเป็น null ให้ใช้ค่า 0.0
    }

    if (order['totalPrice'] is double) {
      return order['totalPrice']; // ถ้าเป็น double ใช้ได้เลย
    }

    if (order['totalPrice'] is int) {
      return order['totalPrice'].toDouble(); // ถ้าเป็น int แปลงเป็น double
    }

    if (order['totalPrice'] is String) {
      return double.tryParse(order['totalPrice']) ??
          0.0; // ถ้าเป็น String แปลงเป็น double
    }

    return 0.0; // ค่าอื่นๆ ใช้ 0.0 ป้องกัน error
  }

  // ฟังก์ชันแปลง Timestamp เป็น DateTime
  String formatDate(Timestamp timestamp) {
    final date =
        DateTime.fromMillisecondsSinceEpoch(timestamp.millisecondsSinceEpoch);
    return "${date.day}/${date.month}/${date.year}";
  }

  @override
  Widget build(BuildContext context) {
    if (user == null) {
      return Scaffold(
        body: Center(child: Text("กรุณาเข้าสู่ระบบก่อน")),
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
              "สถานะคำสั่งซื้อ",
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
        child: StreamBuilder(
          stream: FirebaseFirestore.instance
              .collection('Orders')
              .where('userId', isEqualTo: user!.uid) // กรอง userId
              .where('status', whereIn: [
            'Waiting',
            'Accepted',
            'Completed',
            'Delivered'
          ]) // กรอง status
              .snapshots(),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.data!.docs.isEmpty) {
              return const Center(child: Text('ไม่มีคำสั่งซื้อของคุณ'));
            }

            final orders = snapshot.data!.docs;
            return ListView.builder(
              itemCount: orders.length,
              itemBuilder: (context, index) {
                final order = orders[index].data() as Map<String, dynamic>;
                final orderId = orders[index].id;
                final items = order['items'] as List;
                final status = order['status'];

                // ✅ ซ่อนออเดอร์ที่มีสถานะ "Payment Completed"
                if (status == 'Payment Completed') {
                  return SizedBox.shrink();
                }

                // // ✅ เรียกป๊อปอัปเมื่อออเดอร์เป็น "Delivered" และยังไม่มีสลิป
                // if (status == 'Delivered' &&
                //     !order.containsKey('paymentSlip')) {
                //   Future.microtask(() {
                //     _showUploadSlipDialog(orderId);
                //   });
                // }

                return Card(
                  margin:
                      const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  child: ListTile(
                    title: Text('Order ID: $orderId'),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Text(
                              'สถานะ: ',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            Text(
                              'สถานะ: ${translateStatus(status)}',
                              style: TextStyle(
                                color: status == 'Waiting'
                                    ? Colors.orange
                                    : (status == 'Accepted'
                                        ? Colors.blue
                                        : (status == 'Delivered'
                                            ? Colors.green
                                            : Colors.black)),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        Text(
                          'วันที่: ${order['timestamp'] != null ? formatDate(order['timestamp']) : 'N/A'}',
                          style: const TextStyle(fontSize: 14),
                        ),
                        const Text('รายการอาหาร:'),
                        ...List.generate(
                          items.length,
                          (i) {
                            final item = items[i];
                            final price =
                                double.tryParse(item['price'].toString()) ??
                                    0.0;
                            final quantity = item['quantity'] ?? 1;
                            final totalItemPrice = price * quantity;

                            // ดึงรายการ sides ถ้ามี
                            final List<dynamic>? sides =
                                item['sides'] as List<dynamic>?;

                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                    '- ${item['name']} x$quantity (${totalItemPrice.toInt()})'),
                                if (sides != null && sides.isNotEmpty) ...[
                                  const Text('  เพิ่มเติม :',
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold)),
                                  ...sides.map((side) {
                                    final sideName =
                                        side['name'] ?? 'ไม่มีชื่อ';
                                    final sidePrice = double.tryParse(
                                            side['price'].toString()) ??
                                        0.0;
                                    return Text(
                                        '    - $sideName (+${sidePrice.toInt()})');
                                  }).toList(),
                                ],
                              ],
                            );
                          },
                        ),
                        Text(
                          'ยอดรวม: ${_getTotalPrice(order).toInt()} บาท',
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Color.fromARGB(255, 21, 184, 75),
                          ),
                        ),
                      ],
                    ),
                    trailing: Row(
                      mainAxisSize: MainAxisSize
                          .min, // ป้องกันไม่ให้ Row กินพื้นที่เกินไป
                      children: [
                        if (order['status'] == 'Waiting')
                          ElevatedButton(
                            onPressed: () {
                              showCancelOrderDialog(context, orderId);
                            },
                            child: const Text(
                              'ยกเลิกคำสั่งซื้อ',
                              style: TextStyle(color: Colors.white),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.red,
                            ),
                          ),
                        // if (order['status'] == 'Delivered' &&
                        //     !order.containsKey('paymentSlip'))
                        //   ElevatedButton.icon(
                        //     onPressed: () => _uploadPaymentSlip(orderId),
                        //     icon: const Icon(Icons.upload, color: Colors.white),
                        //     label: const Text("อัปโหลดสลิป"),
                        //     style: ElevatedButton.styleFrom(
                        //       backgroundColor: Colors.blue,
                        //       shape: RoundedRectangleBorder(
                        //           borderRadius: BorderRadius.circular(30)),
                        //     ),
                        //   ),
                      ],
                    ),
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
}
