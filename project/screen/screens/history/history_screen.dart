import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class HistoryScreen extends StatefulWidget {
  @override
  _HistoryScreenState createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    if (user == null) {
      return Scaffold(
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
              "ประวัติออเดอร์",
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
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance.collection('Orders').snapshots(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text("เกิดข้อผิดพลาด: ${snapshot.error}"));
          }
          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return const Center(child: Text("ไม่มีประวัติออเดอร์"));
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
              final date = order['timestamp'] != null
                  ? (order['timestamp'] as Timestamp).toDate()
                  : DateTime.now();

              String formattedDate =
                  DateFormat('dd/MM/yyyy HH:mm').format(date); // ✅ รูปแบบวันที่

              return status == 'Payment Completed' // ✅ กรองเฉพาะสถานะ Payment Completed
                  ? Card(
                      elevation: 3,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: ExpansionTile(
                        leading: const Icon(Icons.receipt_long_outlined,
                            color: Colors.orange),
                        title: Text(
                          "Order ID: $orderId",
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "สถานะ: สำเร็จ",
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.green,
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
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold)),
                                  ...item['sides'].map<Widget>((side) => Text(
                                        "- ${side['name']} (+${side['price']} บาท)",
                                      )),
                                ],

                                // ✅ แสดงตัวเลือกพิเศษ (ถ้ามี)
                                if (item['specials'] != null &&
                                    (item['specials'] as List).isNotEmpty) ...[
                                  const Text("ตัวเลือกพิเศษ:",
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold)),
                                  ...item['specials']
                                      .map<Widget>((special) => Text(
                                            "- ${special['name']} (+${special['price']} บาท)",
                                          )),
                                ],

                                // ✅ แสดงหมายเหตุ (ถ้ามี)
                                if (item['note'] != null &&
                                    item['note'].toString().isNotEmpty)
                                  Text("หมายเหตุ: ${item['note']}",
                                      style: const TextStyle(
                                          color: Colors.black87)),
                              ],
                            ),
                          );
                        }).toList(),
                      ),
                    )
                  : const SizedBox
                      .shrink(); // ✅ ไม่แสดงอะไรเลยหากสถานะไม่ใช่ Payment Completed
            },
          );
        },
      ),
    );
  }
}
