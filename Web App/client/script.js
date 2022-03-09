$(document).ready(function(){
    $('#predict').click(function(){
        var name=$('#name').val()
        var transmission = $('#transmission').val()
        var fuel = $('#fuel').val()
        var owner = $('#owner').val()
        var year = $('#year').val()
        var km_driven = $('#km_driven').val().trim()
        var engine = $('#engine').val().trim()
        var max_power = $('#max_power').val().trim()
        if(km_driven&&engine&&max_power){
           //if((parseInt(km_driven) >=0 && parseInt(km_driven)<=200000)&&(parseInt(engine) >=500 && parseInt(engine)<=4000)&&(parseInt(max_power)>=10 && mparseInt(max_power)<=300)){
                $('#wait').html(
                    '<button class="btn btn-info">'+
                        '<span class="spinner-border spinner-border-lg"></span>'+
                        'Estimation du prix. S"il vous plaît, attendez.'+
                    '</button>'
                );
                $('#fill1').text('');
                $('#fill2').text('');
                $('#fill3').text('');
                $.post('/predict',{
                    name:name,
                    transmission: transmission,
                    fuel: fuel,
                    owner: owner,
                    year: year,
                    km_driven: km_driven,
                    engine:engine,
                    max_power:max_power
                },function(data,status){
                    console.log(status);
                    $('#wait').html('');
                    $('#price').val("Php "+data['estimated_price']);
                })
            
        }else{
            if(km_driven==="")
            {
                $('#fill1').text('Veuillez remplir ce champ.');
            }
            if(engine==="")
            {
                $('#fill2').text('Veuillez remplir ce champ.');
            }
            if(engine==="")
            {
                $('#fill3').text('Veuillez remplir ce champ.');
            }
            
            
            
        }

    });
});
function onPageLoad() {
    var url = "/get_car_name"; 
    $.get(url,function(data, status) {
        if(data) {
            var car_name = data.car_name;
            $('#name').empty();
            for(var i in car_name) {
                var opt = new Option(car_name[i]);
                $('#name').append(opt);
            }
        }
    });
    for (var var1 = 2006; var1 <= 2021; var1++)
    {
        var opt1 = new Option(var1);
        $('#year').append(opt1); 
    }
  }
  window.onload = onPageLoad;
