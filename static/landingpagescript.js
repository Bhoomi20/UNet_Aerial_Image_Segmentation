$(document).ready(function(){
    $('.loading').hide();
    var scrollTop = 0;
    $(window).scroll(function(){
      scrollTop = $(window).scrollTop();
       $('.counter').html(scrollTop);
      
      if (scrollTop >= 100) {
        $('#global-nav').addClass('scrolled-nav');
      } else if (scrollTop < 100) {
        $('#global-nav').removeClass('scrolled-nav');
      } 
      
    }); 
    
  });

  function showSpinner()
  {
    $('.loading').show();
  }
  //"{{ url_for('static', filename='main.css') }}