<?php
$servername = "localhost";
$username = "root";
$password = "root";
$db_name = "student_attend";
$conn = new mysqli($servername, $username, $password, $db_name, 4307);
if ($conn-> connect_error){
    die("connection failed.$conn -> connect_error");
}
echo "";
?>