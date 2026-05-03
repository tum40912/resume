import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';
import 'package:krua_pa_ree/screens/review/review_screen.dart';


class OrderHistoryScreen extends StatefulWidget {
  @override
  _OrderHistoryScreenState createState() => _OrderHistoryScreenState();
}

class _OrderHistoryScreenState extends State<OrderHistoryScreen> {
  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    if (user == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text("ประวัติคำสั่งซื้อ"),
          backgroundColor: Colors.orange,
        ),
        body: const Center(child: Text("กรุณาเข้าสู่ระบบก่อน")),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("ประวัติคำสั่งซื้อ"),
        backgroundColor: Colors.orangeAccent,
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance
            .collection('Orders')
            .where('userId', isEqualTo: user.uid)
            .where('status', whereIn: ['Payment Completed', 'Cancelled'])
            .orderBy('timestamp', descending: true) // ✅ เรียงจากล่าสุด
            .snapshots(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text("เกิดข้อผิดพลาด: ${snapshot.error}"));
          }
          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(child: Text("ไม่มีประวัติคำสั่งซื้อ"));
          }

          final orders = snapshot.data!.docs;

          return ListView.builder(
            padding: const EdgeInsets.all(16.0),
            itemCount: orders.length,
            itemBuilder: (context, index) {
              final order = orders[index].data() as Map<String, dynamic>;
              final orderId = orders[index].id;
              final status = order['status'] ?? 'ไม่มีสถานะ';
              final totalPrice = order['totalPrice'] ?? 0;
              final items = order['items'] as List<dynamic>? ?? [];
              final date = order['date'] != null
                  ? (order['date'] as Timestamp).toDate()
                  : DateTime.now();

              String formattedDate =
                  DateFormat('dd/MM/yyyy HH:mm').format(date); // ✅ รูปแบบวันที่

              return Card(
                elevation: 3,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                child: ExpansionTile(
                  leading: const Icon(Icons.receipt_long_outlined, color: Colors.orange), // ✅ ใส่ไอคอนใบเสร็จ
                  title: Text(
                    "Order ID: $orderId",
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "สถานะ: ${status == 'Payment Completed' ? 'สำเร็จ' : 'ยกเลิก'}",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color:
                              status == 'Payment Completed' ? Colors.green : Colors.red,
                        ),
                      ),
                      Text("ยอดรวม: ฿$totalPrice"),
                      Text("วันที่: $formattedDate"),
                    ],
                  ),
                  children: items.map((item) {
                    return ListTile(
                      title: Text(item['name'] ?? "ไม่มีชื่อ"),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("จำนวน: ${item['quantity']}"),
                          
                          // ✅ แสดงเครื่องเคียง (ถ้ามี)
                          if (item['sides'] != null &&
                              (item['sides'] as List).isNotEmpty) ...[
                            const Text("เครื่องเคียง:",
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            ...item['sides'].map<Widget>((side) => Text(
                                  "- ${side['name']} (+${side['price']} บาท)",
                                )),
                          ],

                          // ✅ แสดงตัวเลือกพิเศษ (ถ้ามี)
                          if (item['specials'] != null &&
                              (item['specials'] as List).isNotEmpty) ...[
                            const Text("ตัวเลือกพิเศษ:",
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            ...item['specials'].map<Widget>((special) => Text(
                                  "- ${special['name']} (+${special['price']} บาท)",
                                )),
                          ],

                          // ✅ แสดงหมายเหตุ (ถ้ามี)
                          if (item['note'] != null &&
                              item['note'].toString().isNotEmpty)
                            Text("หมายเหตุ: ${item['note']}",
                                style: const TextStyle(color: Colors.black87)),
                        ],
                      ),
                    );
                  }).toList(),

                  // ✅ ปุ่มให้คะแนนรีวิว
                  trailing: status == 'Payment Completed' &&
                          !(order.containsKey('review') &&
                              order['review'] != null)
                      ? ElevatedButton.icon(
                          onPressed: () async {
                            final result = await Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => ReviewScreen(
                                  orderId: orderId,
                                ),
                              ),
                            );

                            // หากรีวิวสำเร็จ ให้รีเฟรช UI
                            if (result == true) {
                              setState(() {});
                            }
                          },
                          icon: const Icon(Icons.star_rate, color: Colors.white),
                          label: const Text("ให้คะแนน"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.amber,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                            elevation: 5,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 8,
                            ),
                            textStyle: const TextStyle(fontSize: 14),
                          ),
                        )
                      : null,
                ),
              );
            },
          );
        },
      ),
    );
  }
}
