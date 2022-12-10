# create api view for excecute machine learning model
# include feature selection and algorithm selection
from rest_framework.views import APIView
from rest_framework.response import Response
from machineLearning import CardioML


class CardioMLView(APIView):
    cardioMl = CardioML()

    def post(self, request, *args, **kwargs):
        # get data from request
        features = request.data['features']
        algos = request.data['algos']
        test_size = request.data['test_size']
        result = self.cardioMl.execute_ml(algos, features, test_size)
        # response ok
        return Response(result, status=200)

    def get(self, request, *args, **kwargs):
        # tra ve ma tran tuong quan
        corr_mat = self.cardioMl.get_corr()
        return Response({
            "features": corr_mat
        }, status=200)
