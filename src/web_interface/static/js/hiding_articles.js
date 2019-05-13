$(document).ready(function(){
    $(".article_1_link").click(function(){
        $("#article1").show();
        $("#article2").hide();
        $("#article3").hide();


    });

    $(".article_2_link").click(function(){
        $("#article2").show();
        $("#article1").hide();
        $("#article3").hide();

    });

    $(".article_3_link").click(function(){
        $("#article3").show();
        $("#article1").hide();
        $("#article2").hide();


    });
  });

