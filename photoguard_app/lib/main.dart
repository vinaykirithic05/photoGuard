import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:animate_do/animate_do.dart';

void main() {
  runApp(const PhotoGuardApp());
}

class PhotoGuardApp extends StatelessWidget {
  const PhotoGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PhotoGuard AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          secondary: Color(0xFF10B981),
          surface: Color(0xFF1E293B),
        ),
        textTheme: GoogleFonts.outfitTextTheme(ThemeData.dark().textTheme),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;
  Uint8List? _gradcamBytes;
  final ImagePicker _picker = ImagePicker();

  // Change this to your server IP if testing on a physical mobile device (e.g. http://192.168.1.X:8000)
  final String _apiUrl = "http://127.0.0.1:8000/api/v1/predict";

  Future<void> _pickImage(ImageSource source) async {
    final XFile? pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        _selectedImageName = pickedFile.name;
        _predictionResult = null;
        _gradcamBytes = null;
      });
      _analyzePhoto();
    }
  }

  Future<void> _analyzePhoto() async {
    if (_selectedImageBytes == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          _selectedImageBytes!,
          filename: _selectedImageName ?? 'uploaded_photo.jpg',
        ),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          _predictionResult = data;
          if (data['gradcam_base64'] != null) {
            _gradcamBytes = base64Decode(data['gradcam_base64']);
          }
        });
      } else {
        _showErrorSnackBar("Server Error: ${response.statusCode}");
      }
    } catch (e) {
      _showErrorSnackBar("Connection Failed. Ensure FastAPI is running on port 8000.");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.redAccent,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    bool isAi = _predictionResult?['prediction']?.toString().toLowerCase() == 'ai';

    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E293B),
        elevation: 0,
        centerTitle: true,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.shield_outlined, color: Color(0xFF6366F1)),
            const SizedBox(width: 8),
            Text(
              "PhotoGuard AI",
              style: GoogleFonts.outfit(
                fontWeight: FontWeight.bold,
                fontSize: 22,
              ),
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Container
            Container(
              height: 320,
              decoration: BoxDecoration(
                color: const Color(0xFF1E293B),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: _predictionResult == null
                      ? Colors.white12
                      : (isAi ? Colors.redAccent : const Color(0xFF10B981)),
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.3),
                    blurRadius: 15,
                    offset: const Offset(0, 5),
                  )
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(18),
                child: _selectedImageBytes == null
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.add_photo_alternate_outlined,
                            size: 64,
                            color: Colors.grey.shade600,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            "Select or Capture a Photo",
                            style: TextStyle(
                              color: Colors.grey.shade400,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      )
                    : Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
                          if (_isLoading)
                            Container(
                              color: Colors.black54,
                              child: const Center(
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    CircularProgressIndicator(
                                      color: Color(0xFF6366F1),
                                    ),
                                    SizedBox(height: 16),
                                    Text(
                                      "Analyzing Image Patterns...",
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    )
                                  ],
                                ),
                              ),
                            ),
                        ],
                      ),
              ),
            ),

            const SizedBox(height: 24),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text("Gallery"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF6366F1),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text("Camera"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1E293B),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 28),

            // Prediction Results Card
            if (_predictionResult != null) ...[
              FadeInUp(
                duration: const Duration(milliseconds: 500),
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E293B),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Icon(
                                isAi ? Icons.warning_amber_rounded : Icons.verified_user,
                                color: isAi ? Colors.redAccent : const Color(0xFF10B981),
                                size: 32,
                              ),
                              const SizedBox(width: 12),
                              Text(
                                isAi ? "AI-GENERATED" : "REAL PHOTO",
                                style: GoogleFonts.outfit(
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold,
                                  color: isAi ? Colors.redAccent : const Color(0xFF10B981),
                                ),
                              ),
                            ],
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: isAi
                                  ? Colors.redAccent.withValues(alpha: 0.2)
                                  : const Color(0xFF10B981).withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              "${_predictionResult!['confidence']}% Confident",
                              style: TextStyle(
                                color: isAi ? Colors.redAccent : const Color(0xFF10B981),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          )
                        ],
                      ),
                      const Divider(height: 24, color: Colors.white12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          _buildProbIndicator(
                            "AI Probability",
                            "${_predictionResult!['probabilities']?['AI'] ?? _predictionResult!['probabilities']?['ai'] ?? 0}%",
                            Colors.redAccent,
                          ),
                          _buildProbIndicator(
                            "Real Probability",
                            "${_predictionResult!['probabilities']?['Real'] ?? _predictionResult!['probabilities']?['real'] ?? 0}%",
                            const Color(0xFF10B981),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // Grad-CAM Explainability Visual Heatmap
              if (_gradcamBytes != null) ...[
                FadeInUp(
                  duration: const Duration(milliseconds: 600),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.remove_red_eye_outlined, color: Color(0xFF6366F1)),
                            const SizedBox(width: 8),
                            Text(
                              "Grad-CAM Explainability Heatmap",
                              style: GoogleFonts.outfit(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.memory(
                            _gradcamBytes!,
                            fit: BoxFit.cover,
                            height: 250,
                            width: double.infinity,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Red/Orange areas highlight the key pixels used by the CNN to make this prediction.",
                          style: TextStyle(
                            color: Colors.grey.shade400,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildProbIndicator(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(color: Colors.grey.shade400, fontSize: 13),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.outfit(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }
}
