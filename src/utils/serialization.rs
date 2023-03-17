use crate::rcn::{Bias, Weights};
use nalgebra::{DMatrix, DVector};
use serde::{
    de::{self, Deserializer, MapAccess, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, SerializeStruct},
    Deserialize,
};

/// Serialize the matrices into a temporary data structure represented with a dimension field and a
/// data field.
impl Serialize for Weights {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Weights", 1)?;

        // (usize, usize)
        state.serialize_field("dims", &self.0.shape())?;

        // Vec<f64>
        state.serialize_field("data", &self.0.iter().collect::<Vec<_>>())?;

        state.end()
    }
}

/// Deserialize weights into a temporary SerializedWeights type and then construct the matrices from
/// there.
impl<'de> Deserialize<'de> for Weights {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SerializedWeights {
            dims: (usize, usize),
            data: Vec<f64>,
        }

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Dims,
            Data,
        }

        struct SerializedWeightsVisitor;
        impl<'de> Visitor<'de> for SerializedWeightsVisitor {
            type Value = SerializedWeights;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("SerializedWeights")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let dims = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let data = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(SerializedWeights { dims, data })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let (mut dims, mut data) = (None, None);
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Dims => {
                            if dims.is_some() {
                                return Err(de::Error::duplicate_field("dims"));
                            }
                            dims = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    };
                }

                let dims = dims.ok_or_else(|| de::Error::missing_field("dims"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                Ok(SerializedWeights { dims, data })
            }
        }

        const FIELDS: &[&str] = &["dims", "data"];
        let w = deserializer.deserialize_struct("Weights", FIELDS, SerializedWeightsVisitor)?;
        Ok(Weights(DMatrix::from_vec(w.dims.0, w.dims.1, w.data)))
    }
}

/// Treat Bias vectors as sequences that can be collected into a Vec<f64> and later converted into
/// a DVector<f64>.
impl Serialize for Bias {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.shape().0))?;
        for e in &self.0 {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

/// Treat Bias vectors as sequences that can be collected into a Vec<f64> and later converted into
/// a DVector<f64>.
impl<'de> Deserialize<'de> for Bias {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DVectorVisitor;
        impl<'de> Visitor<'de> for DVectorVisitor {
            type Value = Vec<f64>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("DVector<f64>/Vec<f64>")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                match seq.size_hint() {
                    Some(size) => {
                        let mut v = Vec::with_capacity(size);
                        while let Some(x) = seq.next_element()? {
                            v.push(x);
                        }
                        Ok(v)
                    }
                    None => Ok(Vec::new()),
                }
            }
        }
        let v: Vec<f64> = deserializer.deserialize_seq(DVectorVisitor)?;
        Ok(Bias(DVector::from_vec(v)))
    }
}
