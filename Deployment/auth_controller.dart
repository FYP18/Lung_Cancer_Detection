import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:cancer_hist_analyzer/pages/login_page.dart';
import 'package:cancer_hist_analyzer/pages/profile_page.dart';



class AuthController extends GetxController{
  static AuthController instance = Get.find();
  late Rx<User?> _user;

  FirebaseAuth auth = FirebaseAuth.instance;
  @override
  void onReady(){
    super.onReady();
    _user = Rx<User?> (auth.currentUser );
    _user.bindStream(auth.userChanges());
    ever(_user, _initialscreen);
  }

  _initialscreen(User? user){
    if(user==null){
      print("Login Page");
      Get.offAll(()=>const LoginPage());

    }else {
      Get.offAll(()=>const ProfilePage());
    }
  }

  void  register(String email, password) {
    try {
      auth.createUserWithEmailAndPassword(email: email, password: password);
    } catch (e) {
      Get.snackbar("About User", "User Message",
        backgroundColor: Colors.redAccent,
        snackPosition: SnackPosition.BOTTOM,
        titleText: Text
          (
          "Account Creation failed",
          style : TextStyle(
              color:Colors.white
          ),
        ),
      );
    }
  }}
